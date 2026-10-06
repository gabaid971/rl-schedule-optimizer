import time

class Revenue:
    def __init__(self, current, lambdas, min_connection_time=60, max_connection_time=480):
        self.current = current
        self.lambdas = lambdas
        self.min_connection_time = min_connection_time
        self.max_connection_time = max_connection_time
        
        # Timing instrumentation
        self.timers = {
            'polars_computation': 0.0,
            'numpy_computation': 0.0,
            'total_computation': 0.0
        }
        self.call_count = 0
    
    def calculate_revenue(self):
        calc_start = time.perf_counter()
        self.call_count += 1
        
        connections = self.current['connections']
        try:
            polars_start = time.perf_counter()
            import polars as pl
            if isinstance(connections, pl.DataFrame):
                df = connections.select(['leg1', 'leg2', 'cnx_time'])
                # Filter feasible cnx_time
                df = df.filter((pl.col('cnx_time') >= self.min_connection_time) & (pl.col('cnx_time') <= self.max_connection_time))
                if df.height == 0:
                    calc_end = time.perf_counter()
                    self.timers['polars_computation'] += calc_end - polars_start
                    self.timers['total_computation'] += calc_end - calc_start
                    return 0.0

                df = df.with_columns((pl.col('leg1') + pl.col('leg2')).alias('pair'))
                lambdas_items = list(self.lambdas.items())
  
                lambdas_df = pl.DataFrame({
                    'pair': [k for k, _ in lambdas_items],
                    'Lambda': [v for _, v in lambdas_items]
                })
                df = df.join(lambdas_df, on='pair', how='left').with_columns(pl.col('Lambda').fill_null(0.0))
                # df = df.with_columns(pl.col('pair').map_dict(self.lambdas, default=0.0).alias('Lambda'))

                cnx = pl.col('cnx_time')
                preference_expr = (
                    pl.when((cnx >= self.min_connection_time) & (cnx <= 120))
                    .then(0.5 + 0.5 * (cnx - self.min_connection_time) / (120 - self.min_connection_time))
                    .when((cnx > 120) & (cnx <= self.max_connection_time))
                    .then(1 - 0.5 * (cnx - 120) / (self.max_connection_time - 120))
                    .otherwise(0.0)
                )

                df = df.with_columns(preference_expr.alias('preference_value'))
                df = df.with_columns((pl.col('Lambda') * pl.col('preference_value')).alias('revenue'))
                total = df.select(pl.col('revenue').sum()).to_numpy()[0][0]
                
                polars_end = time.perf_counter()
                self.timers['polars_computation'] += polars_end - polars_start
                calc_end = time.perf_counter()
                self.timers['total_computation'] += calc_end - calc_start
                
                return float(total)
        except Exception:
            pass

        # Fallback / numpy path (iterative)
        numpy_start = time.perf_counter()
        total_revenue = 0.0
        for connection in connections:
            try:
                cnx_time = float(connection[4])
            except Exception:
                continue
            if cnx_time >= self.min_connection_time and cnx_time <= self.max_connection_time:
                leg1 = connection[2]
                leg2 = connection[3]
                Lambda = self.lambdas.get(str(leg1) + str(leg2), 0.0)
                preference_value = self.preference_curve(cnx_time)
                cnx_revenue = preference_value * Lambda
                total_revenue += cnx_revenue
                
        numpy_end = time.perf_counter()
        self.timers['numpy_computation'] += numpy_end - numpy_start
        calc_end = time.perf_counter()
        self.timers['total_computation'] += calc_end - calc_start
        
        return total_revenue

    def get_timing_report(self):
        """Get a detailed timing report of revenue calculation performance"""
        if self.call_count == 0:
            return "No revenue calculations completed yet."
            
        report = f"\n=== Revenue Timing Report ===\n"
        report += f"Total calls: {self.call_count}\n"
        report += f"Average calculation time: {self.timers['total_computation'] / self.call_count * 1000:.2f} ms\n\n"
        
        for component, total_time in self.timers.items():
            if component != 'total_computation' and total_time > 0:
                avg_time = total_time / max(self.call_count, 1) * 1000
                pct_of_total = (total_time / self.timers['total_computation'] * 100) if self.timers['total_computation'] > 0 else 0
                report += f"{component:20}: {avg_time:6.2f} ms/call ({pct_of_total:5.1f}% of total)\n"
        
        return report

    def preference_curve(self, x):
        if 60 <= x <= 120:
            return 0.5 + 0.5 * (x - 60) / 60
        elif 120 < x < 480:
            return 1 - 0.5 * (x - 120) / 360
        else:
            return 0
