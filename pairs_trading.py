from collections import deque
import numpy as np
from dataclasses import dataclass, field
import utils
import asyncio
import itertools
import time

# Максимальная длина истории свечей для хранения
MAX_LEN = 500
# Окно для анализа коинтеграции
COINT_WINDOW = 200

@dataclass
class Data:
    """
    Класс для хранения временных рядов в deque для каждого символа.
    """
    ts: deque = field(default_factory=lambda: deque(maxlen=MAX_LEN))
    open: deque = field(default_factory=lambda: deque(maxlen=MAX_LEN))
    high: deque = field(default_factory=lambda: deque(maxlen=MAX_LEN))
    low: deque = field(default_factory=lambda: deque(maxlen=MAX_LEN))
    close: deque = field(default_factory=lambda: deque(maxlen=MAX_LEN))

    def add_kline(self, ts, open_p, high_p, low_p, close_p):
        """
        Добавляет новую свечу, если она не дублируется.
        """
        ts = int(ts)
        if not self.ts or ts > self.ts[-1]:
            self.ts.append(ts)
            self.open.append(float(open_p))
            self.high.append(float(high_p))
            self.low.append(float(low_p))
            self.close.append(float(close_p))
            return True
        return False

@dataclass
class PairInfo:
    """
    Хранит информацию о коинтегрированной паре.
    """
    symbol1: str
    symbol2: str
    hedge_ratio: float = 0.0
    half_life: float = 0.0
    last_z_score: float = 0.0
    position_status: int = 0  # 0: нет позиции, 1: лонг спред, -1: шорт спред
    qty1: float = 0.0
    qty2: float = 0.0

class PairsManager:
    """
    Управляет данными о символах, находит коинтегрированные пары и генерирует сигналы.
    """
    def __init__(self, client, loop, all_symbols, min_data_points=COINT_WINDOW):
        self.client = client
        self.loop = loop
        self.all_symbols = all_symbols
        self.min_data_points = min_data_points
        self.all_data: dict[str, Data] = {}
        self.active_pairs: dict[frozenset, PairInfo] = {}
        self._discovery_task = None
        self._last_discovery_time = 0

    async def add_kline(self, kline_data):
        """
        Обрабатывает новую свечу из вебсокета.
        """
        symbol = kline_data['s']
        
        if symbol not in self.all_data:
            self.all_data[symbol] = Data()
            await self._initialize_history(symbol)

        # Добавляем новую свечу
        added = self.all_data[symbol].add_kline(
            kline_data['t'],
            kline_data['o'],
            kline_data['h'],
            kline_data['l'],
            kline_data['c']
        )

        if added:
            # Запускаем анализ для этого символа
            self.loop.create_task(self.run_analysis(symbol))

    async def _initialize_history(self, symbol):
        """
        Загружает исторические данные для инициализации deques.
        """
        print(f"Initializing history for {symbol}...")
        try:
            # Загружаем последние N свечей. Binance вернет до 1000.
            klines = await self.client.klines(symbol, '1m', limit=MAX_LEN)
            data = self.all_data[symbol]
            for k in klines:
                data.add_kline(k[0], k[1], k[2], k[3], k[4])
            print(f"History for {symbol} initialized with {len(data.ts)} candles.")
        except Exception as e:
            print(f"Error initializing history for {symbol}: {e}")
            # Если не удалось загрузить, удаляем, чтобы попробовать снова
            if symbol in self.all_data:
                del self.all_data[symbol]

    async def run_analysis(self, updated_symbol: str):
        """
        Запускает анализ для пар, включающих обновленный символ.
        """
        # 1. Проверяем сигналы для активных пар
        await self._check_signals_for_active_pairs(updated_symbol)

        # 2. Периодически запускаем поиск новых пар (раз в 10 минут)
        now = time.time()
        if now - self._last_discovery_time > 600:
            # Убеждаемся, что предыдущая задача поиска завершена
            if self._discovery_task is None or self._discovery_task.done():
                self._last_discovery_time = now
                self._discovery_task = self.loop.create_task(self._discover_new_pairs())


    async def _check_signals_for_active_pairs(self, updated_symbol: str):
        """
        Проверяет наличие торговых сигналов для активных пар, включающих обновленный символ.
        """
        Z_ENTRY_THRESHOLD = 2.0 # Порог для входа
        Z_EXIT_THRESHOLD = 0.5 # Порог для выхода

        for pair_set, pair_info in self.active_pairs.items():
            if updated_symbol in pair_set:
                s1, s2 = pair_info.symbol1, pair_info.symbol2
                
                # Убедимся, что оба символа имеют данные
                if s1 not in self.all_data or s2 not in self.all_data:
                    continue
                
                data1 = self.all_data[s1]
                data2 = self.all_data[s2]

                # Убедимся, что данных достаточно
                if len(data1.close) < self.min_data_points or len(data2.close) < self.min_data_points:
                    continue

                # Расчет спреда
                log_prices1 = np.log(list(data1.close)[-self.min_data_points:])
                log_prices2 = np.log(list(data2.close)[-self.min_data_points:])
                spread = log_prices1 - pair_info.hedge_ratio * log_prices2
                
                # Расчет Z-score
                z_score = utils.calculate_z_last(spread)
                if z_score is None:
                    continue
                
                pair_info.last_z_score = z_score

                # --- Логика входа в позицию ---
                # Если позиции нет
                if pair_info.position_status == 0:
                    if z_score < -Z_ENTRY_THRESHOLD:
                        # Сигнал: купить спред (лонг s1, шорт s2)
                        print(f"🚀 LONG Signal on {s1}-{s2} spread. Z-score: {z_score:.2f}. Opening position...")
                        self.loop.create_task(self._execute_trade(pair_info, 1))
                        pair_info.position_status = 1
                    elif z_score > Z_ENTRY_THRESHOLD:
                        # Сигнал: продать спред (шорт s1, лонг s2)
                        print(f"🔥 SHORT Signal on {s1}-{s2} spread. Z-score: {z_score:.2f}. Opening position...")
                        self.loop.create_task(self._execute_trade(pair_info, -1))
                        pair_info.position_status = -1
                
                # --- Логика выхода из позиции ---
                # Если в лонге по спреду и z-score пересек порог выхода
                elif pair_info.position_status == 1 and z_score > -Z_EXIT_THRESHOLD:
                    print(f"EXIT LONG Signal on {s1}-{s2}. Z-score: {z_score:.2f}. Closing position...")
                    self.loop.create_task(self._execute_trade(pair_info, 0)) # 0 - сигнал к закрытию
                    pair_info.position_status = 0
                
                # Если в шорте по спреду и z-score пересек порог выхода
                elif pair_info.position_status == -1 and z_score < Z_EXIT_THRESHOLD:
                    print(f"EXIT SHORT Signal on {s1}-{s2}. Z-score: {z_score:.2f}. Closing position...")
                    self.loop.create_task(self._execute_trade(pair_info, 0)) # 0 - сигнал к закрытию
                    pair_info.position_status = 0

    async def _discover_new_pairs(self):
        """
        Ищет новые коинтегрированные пары среди всех доступных символов.
        """
        print("Starting discovery process for new cointegrated pairs...")
        
        # Собираем символы, у которых достаточно данных
        ready_symbols = [s for s, data in self.all_data.items() if len(data.ts) >= self.min_data_points]
        
        if len(ready_symbols) < 2:
            print("Not enough symbols with sufficient data to find pairs.")
            return

        # Генерируем все возможные пары
        potential_pairs = list(itertools.combinations(ready_symbols, 2))
        
        tasks = []
        for pair in potential_pairs:
            pair_set = frozenset(pair)
            # Не проверяем заново уже активные пары
            if pair_set not in self.active_pairs:
                tasks.append(self.loop.create_task(self._analyze_pair(pair[0], pair[1])))
        
        if tasks:
            await asyncio.gather(*tasks)
        print("Discovery process finished.")

    async def _analyze_pair(self, symbol1: str, symbol2: str):
        """
        Анализирует одну пару на коинтеграцию.
        Если пара коинтегрирована, добавляет ее в self.active_pairs.
        """
        data1 = self.all_data.get(symbol1)
        data2 = self.all_data.get(symbol2)

        if not data1 or not data2 or len(data1.ts) < self.min_data_points or len(data2.ts) < self.min_data_points:
            return

        # Убедимся, что данные выровнены по времени (упрощенно, просто берем последние N точек)
        log_prices1 = np.log(list(data1.close)[-self.min_data_points:])
        log_prices2 = np.log(list(data2.close)[-self.min_data_points:])

        flag, hedge, hl, pval = utils.calculate_cointegration(log_prices1, log_prices2)

        if flag == 1:
            pair_set = frozenset([symbol1, symbol2])
            print(f"✅ New cointegrated pair found: {symbol1}-{symbol2} | Half-life: {hl}, p-value: {pval:.4f}")
            self.active_pairs[pair_set] = PairInfo(symbol1=symbol1, symbol2=symbol2, hedge_ratio=hedge, half_life=hl)

    async def _execute_trade(self, pair_info: PairInfo, direction: int):
        """
        Исполняет торговый приказ.
        direction: 1 для лонга спреда, -1 для шорта спреда, 0 для закрытия.
        """
        s1 = pair_info.symbol1
        s2 = pair_info.symbol2
        
        # --- Логика закрытия позиции ---
        if direction == 0:
            if pair_info.position_status == 0:
                return # Нечего закрывать
            
            print(f"EXECUTING CLOSE for {s1}-{s2}")
            side1_close = 'SELL' if pair_info.position_status == 1 else 'BUY'
            side2_close = 'BUY' if pair_info.position_status == 1 else 'SELL'
            qty1_close = pair_info.qty1
            qty2_close = pair_info.qty2

            try:
                task1 = self.loop.create_task(
                    self.client.new_order(symbol=s1, side=side1_close, type='MARKET', quantity=qty1_close, newOrderRespType='RESULT')
                )
                task2 = self.loop.create_task(
                    self.client.new_order(symbol=s2, side=side2_close, type='MARKET', quantity=qty2_close, newOrderRespType='RESULT')
                )
                results = await asyncio.gather(task1, task2, return_exceptions=True)
                
                if any(isinstance(res, Exception) for res in results):
                    print(f"ERROR closing position for {s1}-{s2}. Manual intervention required. Errors: {results}")
                else:
                    print(f"SUCCESS: Position closed for {s1}-{s2}")
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
            except Exception as e:
                print(f"FATAL ERROR closing position for {s1}-{s2}: {e}")
            return

        # --- Логика входа в позицию ---
        hedge = pair_info.hedge_ratio
        s1_info = self.all_symbols.get(s1)
        s2_info = self.all_symbols.get(s2)

        if not s1_info or not s2_info:
            print(f"ERROR: Symbol info not found for {s1} or {s2}")
            pair_info.position_status = 0
            return

        try:
            s1_price = float((await self.client.ticker_price(s1))['price'])
            s2_price = float((await self.client.ticker_price(s2))['price'])
        except Exception as e:
            print(f"ERROR: Could not fetch prices for {s1}/{s2}. E: {e}")
            pair_info.position_status = 0
            return
            
        data1 = self.all_data.get(s1)
        data2 = self.all_data.get(s2)
        log_prices1 = np.log(list(data1.close)[-COINT_WINDOW:])
        log_prices2 = np.log(list(data2.close)[-COINT_WINDOW:])

        dollar1, dollar2 = utils.vol_parity_notional(log_prices1, log_prices2, hedge)
        
        qty1_dollar = dollar1 * direction
        qty2_dollar = dollar2 * -direction

        qty1 = qty1_dollar / s1_price
        qty2 = qty2_dollar / s2_price
        
        side1 = 'BUY' if qty1 > 0 else 'SELL'
        side2 = 'BUY' if qty2 > 0 else 'SELL'

        qty1_rounded = utils.round_down(abs(qty1), s1_info.step_size)
        qty2_rounded = utils.round_down(abs(qty2), s2_info.step_size)

        if qty1_rounded * s1_price < s1_info.notional or qty2_rounded * s2_price < s2_info.notional:
            print(f"WARN: Order size for {s1}-{s2} is below minimum notional. Skipping trade.")
            pair_info.position_status = 0
            return

        print(f"EXECUTING TRADE for {s1}-{s2}:")
        print(f"  {side1} {qty1_rounded} {s1} at {s1_price}")
        print(f"  {side2} {qty2_rounded} {s2} at {s2_price}")

        try:
            task1 = self.loop.create_task(
                self.client.new_order(symbol=s1, side=side1, type='MARKET', quantity=qty1_rounded, newOrderRespType='RESULT')
            )
            task2 = self.loop.create_task(
                self.client.new_order(symbol=s2, side=side2, type='MARKET', quantity=qty2_rounded, newOrderRespType='RESULT')
            )
            
            results = await asyncio.gather(task1, task2, return_exceptions=True)
            
            has_error = False
            executed_orders = []
            for res in results:
                if isinstance(res, Exception):
                    print(f"ERROR placing order: {res}")
                    has_error = True
                else:
                    executed_orders.append(res)
            
            if has_error:
                print("ERROR: One or both orders failed. Manual intervention may be required.")
                # ToDo: Implement recovery logic (e.g., close the successful leg)
                pair_info.position_status = 0
            else:
                pair_info.qty1 = float(executed_orders[0]['executedQty'])
                pair_info.qty2 = float(executed_orders[1]['executedQty'])
                print(f"SUCCESS: Trade executed for {s1}-{s2}. Qty1: {pair_info.qty1}, Qty2: {pair_info.qty2}")
                # ToDo: Save trade details to DB

        except Exception as e:
            print(f"FATAL ERROR during trade execution for {s1}-{s2}: {e}")
            traceback.print_exc()
            pair_info.position_status = 0
