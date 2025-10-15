from binance_bulk_downloader.downloader import BinanceBulkDownloader

# 单一符号
downloader = BinanceBulkDownloader(
    data_type="klines",
    data_frequency="1m",
    asset="spot",
    timeperiod_per_file="daily",
    symbols="ETHUSDT",
)
downloader.run_download()
