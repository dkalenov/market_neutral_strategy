from collections import deque
import numpy as np
from dataclasses import dataclass, field
import utils
import asyncio
import itertools
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
import db

# Максимальная длина истории свечей для хранения
MAX_LEN = 500
# Окно для анализа коинтеграции
COINT_WINDOW = 200

class Data:
    """
    Класс для хранения временных рядов в deque для каждого символа.
    """
    def __init__(self, maxlen=500):
        self.ts = deque(maxlen=maxlen)
        self.open = deque(maxlen=maxlen)
        self.high = deque(maxlen=maxlen)
        self.low = deque(maxlen=maxlen)
        self.close = deque(maxlen=maxlen)

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
    entry_price1: float = 0.0
    entry_price2: float = 0.0
    db_id: int = None # ID записи в БД для синхронизации
    current_trade_id: int = None # ID текущей открытой сделки в таблице Trades

class PairsManager:
    """
    Управляет данными о символах, находит коинтегрированные пары и генерирует сигналы.
    """
    def __init__(self, client, loop, all_symbols, timeframe='1h', min_data_points=200):
        self.client = client
        self.loop = loop
        self.all_symbols = all_symbols
        self.timeframe = timeframe
        self.min_data_points = min_data_points
        self.max_len = int(min_data_points * 2.5) # Храним с запасом
        
        self.all_data: dict[str, Data] = {}
        # Обновляем дефолтные фабрики для Data
        # (На самом деле Data.ts и т.д. создаются при инициализации, так что нужно будет передавать maxlen туда
        #  или просто сделать Data более гибким. Пока оставим жесткий лимит в Data, но он должен быть > min_data_points)
        
        self.active_pairs: dict[frozenset, PairInfo] = {}
        self._discovery_task = None
        self._last_discovery_time = 0
        
        # Пул процессов для тяжелых вычислений (коинтеграция)
        # max_workers=None -> использует все доступные ядра CPU
        self.executor = ProcessPoolExecutor(max_workers=None)
        
        # Загружаем состояние из БД при старте
        self.loop.create_task(self._load_state_from_db())

    async def _load_state_from_db(self):
        print("Restoring state from DB...")
        try:
            pairs = await db.get_all_pairs()
            for p in pairs:
                pair_set = frozenset([p.symbol1, p.symbol2])
                info = PairInfo(
                    symbol1=p.symbol1,
                    symbol2=p.symbol2,
                    hedge_ratio=p.hedge_ratio,
                    half_life=p.half_life,
                    position_status=p.position_status,
                    qty1=p.qty1,
                    qty2=p.qty2,
                    entry_price1=p.entry_price1,
                    entry_price2=p.entry_price2,
                    db_id=p.id
                )
                
                # Если позиция открыта, восстанавливаем ID сделки
                if p.position_status != 0:
                    last_trade = await db.get_last_open_trade_for_pair(p.id)
                    if last_trade:
                        info.current_trade_id = last_trade.id
                        print(f"  Attached open trade ID: {last_trade.id}")
                    else:
                        print(f"  WARN: Position active but no open trade found in DB for pair {p.id}")

                self.active_pairs[pair_set] = info
                print(f"Restored pair {p.symbol1}/{p.symbol2} (Status: {p.position_status}, ID: {p.id})")
        except Exception as e:
            print(f"Error loading state from DB: {e}")

    async def add_kline(self, kline_data):
        """
        Обрабатывает новую свечу из вебсокета.
        """
        symbol = kline_data['s']
        
        if symbol not in self.all_data:
            self.all_data[symbol] = Data(maxlen=self.max_len)
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
            klines = await self.client.klines(symbol, self.timeframe, limit=self.max_len)
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
        Реализует динамический пересчет Hedge Ratio и ротацию пар.
        """
        Z_ENTRY_THRESHOLD = 2.0 # Порог для входа
        Z_EXIT_THRESHOLD = 0.0 # Порог для выхода (Take Profit при возврате к среднему)
        Z_STOP_LOSS = 4.0 # Порог Стоп-Лосса (если раздвижка идет против нас)

        # Создаем копию списка пар, чтобы можно было удалять из словаря во время итерации
        current_pairs = list(self.active_pairs.items())

        for pair_set, pair_info in current_pairs:
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

                # Убедимся, что данные синхронизированы по времени (избегаем рассинхрона вебсокета)
                if data1.ts[-1] != data2.ts[-1]:
                    # Одна из пар отстает. Пропускаем такт.
                    continue

                # Подготовка данных для анализа (логарифмические цены)
                log_prices1 = np.log(list(data1.close)[-self.min_data_points:])
                log_prices2 = np.log(list(data2.close)[-self.min_data_points:])

                # --- ДИНАМИЧЕСКИЙ ПЕРЕСЧЕТ ПАРАМЕТРОВ ---
                # Пересчитываем коинтеграцию на каждом шаге (скользящее окно)
                flag, hedge, hl, pval = utils.calculate_cointegration(log_prices1, log_prices2)

                # --- РОТАЦИЯ ПАР ---
                # Если коинтеграция "сломалась" (flag=0 или hl слишком большой)
                if flag == 0 or hl > 200: # HL > 200 считается слишком медленным возвратом
                    print(f"⚠️ Pair {s1}-{s2} correlation broken (pval: {pval:.4f}, HL: {hl}). Removing...")
                    
                    # Если есть открытая позиция - ЭКСТРЕННОЕ закрытие
                    if pair_info.position_status != 0:
                        print(f"🚨 Force closing position on broken pair {s1}-{s2}!")
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                    
                    # Удаляем пару из БД и памяти
                    if pair_info.db_id:
                        # Логируем причину удаления (BROKEN)
                        reason = f"Broken. Flag={flag}, HL={hl:.2f}, Pval={pval:.4f}"
                        history_item = db.PairHistory(
                            symbol1=s1,
                            symbol2=s2,
                            event_type='BROKEN',
                            timestamp=int(time.time() * 1000),
                            hedge_ratio=hedge,
                            half_life=hl,
                            reason=reason
                        )
                        self.loop.create_task(db.add_pair_history(history_item))

                        await db.delete_pair(pair_info.db_id)
                    
                    if pair_set in self.active_pairs:
                        del self.active_pairs[pair_set]
                    continue

                # Если все ок, обновляем параметры (Динамический Hedge Ratio)
                pair_info.hedge_ratio = hedge
                pair_info.half_life = hl
                
                # Синхронизируем обновление параметров с БД
                if pair_info.db_id:
                    self.loop.create_task(db.update_pair({
                        'id': pair_info.db_id,
                        'hedge_ratio': hedge,
                        'half_life': hl
                    }))

                # Расчет спреда с НОВЫМ hedge_ratio
                spread = log_prices1 - pair_info.hedge_ratio * log_prices2
                
                # Расчет Z-score
                z_score = utils.calculate_z_last(spread)
                if z_score is None:
                    continue
                
                pair_info.last_z_score = z_score

                # --- ЛОГИКА ТОРГОВЛИ ---
                
                # Защитный механизм (Hard PnL Stop Loss)
                # Рассчитываем текущий PnL, если есть позиция
                if pair_info.position_status != 0 and pair_info.entry_price1 > 0 and pair_info.entry_price2 > 0:
                    current_price1 = list(data1.close)[-1]
                    current_price2 = list(data2.close)[-1]
                    
                    # Определяем направление сделки для каждого актива
                    # status 1 (Long Spread): Buy S1, Sell S2
                    # status -1 (Short Spread): Sell S1, Buy S2
                    side1 = 1 if pair_info.position_status == 1 else -1
                    side2 = -1 if pair_info.position_status == 1 else 1

                    pnl1 = (current_price1 - pair_info.entry_price1) * pair_info.qty1 * side1
                    pnl2 = (current_price2 - pair_info.entry_price2) * pair_info.qty2 * side2
                    total_pnl = pnl1 + pnl2
                    
                    initial_investment = (pair_info.entry_price1 * pair_info.qty1) + (pair_info.entry_price2 * pair_info.qty2)
                    
                    # Порог экстренного закрытия: -20% от объема позиции
                    HARD_STOP_PCT = 0.20 
                    
                    if initial_investment > 0:
                        roi = total_pnl / initial_investment
                        # Если просадка больше 20%
                        if roi < -HARD_STOP_PCT:
                            print(f"🚨 CIRCUIT BREAKER TRIGGERED on {s1}-{s2}!")
                            print(f"   Unrealized PnL: {total_pnl:.2f} USD ({roi*100:.2f}%). Force Closing...")
                            self.loop.create_task(self._execute_trade(pair_info, 0))
                            pair_info.position_status = 0
                            continue # Прерываем дальнейшую проверку сигналов

                # 1. Если позиции нет (ВХОД)
                if pair_info.position_status == 0:
                    if z_score < -Z_ENTRY_THRESHOLD:
                        # Z < -2 -> Покупка спреда (Long)
                        print(f"🚀 LONG Signal on {s1}-{s2} spread. Z-score: {z_score:.2f}. Opening position...")
                        self.loop.create_task(self._execute_trade(pair_info, 1))
                        pair_info.position_status = 1
                    elif z_score > Z_ENTRY_THRESHOLD:
                        # Z > 2 -> Продажа спреда (Short)
                        print(f"🔥 SHORT Signal on {s1}-{s2} spread. Z-score: {z_score:.2f}. Opening position...")
                        self.loop.create_task(self._execute_trade(pair_info, -1))
                        pair_info.position_status = -1
                
                # 2. Если мы в ЛОНГЕ (куплен спред)
                elif pair_info.position_status == 1:
                    # Take Profit: Z вернулся к 0 (или выше)
                    if z_score >= Z_EXIT_THRESHOLD:
                        print(f"💰 TAKE PROFIT (Long) on {s1}-{s2}. Z-score: {z_score:.2f} >= 0. Closing...")
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                        pair_info.position_status = 0
                    # Stop Loss: Z упал еще ниже (-4)
                    elif z_score <= -Z_STOP_LOSS:
                        print(f"🛑 STOP LOSS (Long) on {s1}-{s2}. Z-score: {z_score:.2f} <= -4. Closing...")
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                        pair_info.position_status = 0

                # 3. Если мы в ШОРТЕ (продан спред)
                elif pair_info.position_status == -1:
                    # Take Profit: Z вернулся к 0 (или ниже)
                    if z_score <= -Z_EXIT_THRESHOLD:
                        print(f"💰 TAKE PROFIT (Short) on {s1}-{s2}. Z-score: {z_score:.2f} <= 0. Closing...")
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                        pair_info.position_status = 0
                    # Stop Loss: Z вырос еще выше (4)
                    elif z_score >= Z_STOP_LOSS:
                        print(f"🛑 STOP LOSS (Short) on {s1}-{s2}. Z-score: {z_score:.2f} >= 4. Closing...")
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                        pair_info.position_status = 0

    async def _discover_new_pairs(self):
        """
        Ищет новые коинтегрированные пары среди всех доступных символов.
        Использует ПАРАЛЛЕЛЬНЫЕ ВЫЧИСЛЕНИЯ (Multiprocessing) для скорости.
        """
        print("Starting discovery process for new cointegrated pairs (PARALLEL)...")
        start_time = time.time()
        
        # 1. Подготовка снимка данных (snapshot) для передачи в процессы
        # Сериализуем только необходимые данные (log prices), чтобы минимизировать накладные расходы на pickle
        ready_symbols = []
        data_snapshot = {}
        
        for s, data in self.all_data.items():
            if len(data.ts) >= self.min_data_points:
                ready_symbols.append(s)
                # Берем последние min_data_points и логарифмируем сразу здесь
                # Это уменьшает объем передаваемых данных и работу в воркерах
                prices = list(data.close)[-self.min_data_points:]
                data_snapshot[s] = np.log(prices)
        
        if len(ready_symbols) < 2:
            print("Not enough symbols with sufficient data to find pairs.")
            return

        print(f"Analyzing {len(ready_symbols)} symbols using {self.min_data_points} candles.")

        # 2. Генерация всех возможные пар
        # Используем список, так как генератор нельзя разбить на чанки без итерации
        all_combinations = list(itertools.combinations(ready_symbols, 2))
        
        # Фильтруем те, что уже активны
        candidates = [pair for pair in all_combinations if frozenset(pair) not in self.active_pairs]
        
        total_pairs = len(candidates)
        print(f"Total pairs to check: {total_pairs}")
        
        if total_pairs == 0:
            return

        # 3. Разбиение на чанки (Chunking)
        # Оптимальный размер чанка зависит от кол-ва ядер. 
        # 5000 - хороший баланс между накладными расходами IPC и загрузкой ядер.
        CHUNK_SIZE = 5000
        chunks = [candidates[i:i + CHUNK_SIZE] for i in range(0, total_pairs, CHUNK_SIZE)]
        
        print(f"Split into {len(chunks)} chunks for parallel processing.")
        
        # 4. Запуск параллельных задач
        tasks = []
        for chunk in chunks:
            # self.executor распараллелит это по ядрам
            task = self.loop.run_in_executor(
                self.executor, 
                utils.batch_process_pairs, 
                chunk, 
                data_snapshot, 
                self.min_data_points
            )
            tasks.append(task)
        
        # Ждем завершения всех задач
        # results_list будет списком списков результатов
        results_list = await asyncio.gather(*tasks)
        
        # 5. Обработка результатов
        new_pairs_count = 0
        
        for batch_results in results_list:
            for res in batch_results:
                s1, s2, hedge, hl, pval = res
                
                # Добавляем новую пару (логика из старого _analyze_pair)
                try:
                    # Сохраняем в БД
                    new_pair = db.Pairs(
                        symbol1=s1, 
                        symbol2=s2, 
                        hedge_ratio=hedge, 
                        half_life=hl,
                        position_status=0
                    )
                    # Здесь нужен await, поэтому мы не можем делать это внутри process executor
                    await db.add_pair(new_pair)
                    
                    # Логируем историю
                    history_item = db.PairHistory(
                        symbol1=s1, 
                        symbol2=s2, 
                        event_type='FOUND',
                        timestamp=int(time.time() * 1000),
                        hedge_ratio=hedge,
                        half_life=hl,
                        reason='Discovery'
                    )
                    self.loop.create_task(db.add_pair_history(history_item))
                    
                    pair_set = frozenset([s1, s2])
                    print(f"✅ FOUND: {s1}-{s2} | HL: {hl:.2f}, P: {pval:.4f}")
                    
                    # Добавляем в память
                    self.active_pairs[pair_set] = PairInfo(
                        symbol1=s1, 
                        symbol2=s2, 
                        hedge_ratio=hedge, 
                        half_life=hl,
                        db_id=new_pair.id
                    )
                    new_pairs_count += 1
                except Exception as e:
                    print(f"Error adding pair {s1}-{s2}: {e}")

        elapsed = time.time() - start_time
        print(f"Discovery process finished in {elapsed:.2f}s. Found {new_pairs_count} new pairs.")

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
                    # Успешное закрытие
                    print(f"SUCCESS: Position closed for {s1}-{s2}")
                    
                    # Парсим цены выхода и считаем PnL
                    def get_price(order):
                        if 'avgPrice' in order and float(order['avgPrice']) > 0:
                            return float(order['avgPrice'])
                        if 'cummulativeQuoteQty' in order and 'executedQty' in order and float(order['executedQty']) > 0:
                            return float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                        return 0.0

                    close_price1 = get_price(results[0])
                    close_price2 = get_price(results[1])
                    
                    # Расчет PnL
                    # Long Spread (pos=1): Long S1, Short S2. Exit: Sell S1, Buy S2.
                    # PnL1 = (Close - Entry) * Qty
                    # Short Spread (pos=-1): Short S1, Long S2. Exit: Buy S1, Sell S2.
                    # PnL1 = (Entry - Close) * Qty (или (Close - Entry) * Qty * -1)
                    
                    side1_dir = 1 if pair_info.position_status == 1 else -1
                    side2_dir = -1 if pair_info.position_status == 1 else 1

                    pnl1 = (close_price1 - pair_info.entry_price1) * pair_info.qty1 * side1_dir
                    pnl2 = (close_price2 - pair_info.entry_price2) * pair_info.qty2 * side2_dir
                    total_pnl = pnl1 + pnl2
                    
                    print(f"  Realized PnL: {total_pnl:.2f} (S1: {pnl1:.2f}, S2: {pnl2:.2f})")

                    # Закрываем сделку в БД
                    if pair_info.current_trade_id:
                        self.loop.create_task(db.update_trade_fields(
                            pair_info.current_trade_id, 
                            status='CLOSED',
                            close_time=int(time.time() * 1000),
                            close_price_1=close_price1,
                            close_price_2=close_price2,
                            pnl=total_pnl,
                            # Duration (optional calc)
                        ))
                    
                    pair_info.current_trade_id = None
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    
                    # Обновляем состояние пары в БД (сброс)
                    if pair_info.db_id:
                        self.loop.create_task(db.update_pair({
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0
                        }))
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
                print("ERROR: One or both orders failed. Initiating EMERGENCY REVERT for executed legs...")
                
                # Попытка откатить (закрыть) те сделки, которые прошли успешно
                for executed in executed_orders:
                    try:
                        exec_symbol = executed['symbol']
                        exec_qty = float(executed['executedQty'])
                        exec_side = executed['side'] # 'BUY' or 'SELL'
                        
                        # Invert side for close
                        revert_side = 'SELL' if exec_side == 'BUY' else 'BUY'
                        
                        print(f"  Reverting {exec_symbol}: {revert_side} {exec_qty}...")
                        self.loop.create_task(
                             self.client.new_order(symbol=exec_symbol, side=revert_side, type='MARKET', quantity=exec_qty)
                        )
                    except Exception as rev_e:
                        print(f"  CRITICAL: Failed to revert {exec_symbol}: {rev_e}")

                pair_info.position_status = 0
            else:
                pair_info.qty1 = float(executed_orders[0]['executedQty'])
                pair_info.qty2 = float(executed_orders[1]['executedQty'])
                
                # Получаем среднюю цену входа
                # API Binance возвращает 'avgPrice' или можно посчитать через cummulativeQuoteQty / executedQty
                def get_price(order):
                    if 'avgPrice' in order and float(order['avgPrice']) > 0:
                        return float(order['avgPrice'])
                    if 'cummulativeQuoteQty' in order and 'executedQty' in order and float(order['executedQty']) > 0:
                        return float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                    return 0.0 # Fallback

                pair_info.entry_price1 = get_price(executed_orders[0])
                pair_info.entry_price2 = get_price(executed_orders[1])

                print(f"SUCCESS: Trade executed for {s1}-{s2}.")
                print(f"  Entry Prices: {s1}={pair_info.entry_price1}, {s2}={pair_info.entry_price2}")
                print(f"  Quantities: {s1}={pair_info.qty1}, {s2}={pair_info.qty2}")
                
                # Обновляем состояние пары в БД
                if pair_info.db_id:
                    self.loop.create_task(db.update_pair({
                        'id': pair_info.db_id,
                        'position_status': pair_info.position_status,
                        'qty1': pair_info.qty1,
                        'qty2': pair_info.qty2,
                        'entry_price1': pair_info.entry_price1,
                        'entry_price2': pair_info.entry_price2
                    }))

                # Сохраняем сделку в историю (Trades)
                try:
                    trade = db.Trades(
                        pair_id=pair_info.db_id,
                        symbol1=s1,
                        symbol2=s2,
                        direction=direction,
                        status='OPEN',
                        open_time=int(time.time() * 1000),
                        entry_price_1=pair_info.entry_price1,
                        entry_price_2=pair_info.entry_price2,
                        qty1=pair_info.qty1,
                        qty2=pair_info.qty2,
                        pnl=0.0
                    )
                    # Await directly to get the ID
                    pair_info.current_trade_id = await db.add_trade(trade)
                    print(f"  Trade logged to DB with ID: {pair_info.current_trade_id}")
                except Exception as e:
                     print(f"Error creating trade record: {e}")
                
        except Exception as e:
            print(f"FATAL ERROR during trade execution for {s1}-{s2}: {e}")
            traceback.print_exc()
            pair_info.position_status = 0
