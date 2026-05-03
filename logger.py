import requests
import time
import csv

# 配置参数
URL = "http://localhost:8111/state"
FREQ = 20.0
INTERVAL = 1.0 / FREQ

# 需要抓取的目标数据键名
KEYS_TO_EXTRACT = [ 
    "aileron, %", "elevator, %", "rudder, %", "flaps, %",
    "gear, %", "airbrake, %", "H, m", "TAS, km/h", "IAS, km/h",
    "M", "AoA, deg", "AoS, deg", "Ny", "Vy, m/s", "Wx, deg/s",
    "throttle 1, %"
]

# 表头：时间戳 + 数据
HEADER = ["timestamp"] + KEYS_TO_EXTRACT

def main():
    # 动态生成文件名
    current_time_str = time.strftime('%Y%m%d_%H%M%S')
    csv_filename = f"flight_data_{current_time_str}.csv"

    print(f"已启动战争雷霆数据采集程序。")
    print(f"目标端口: {URL} | 采样率: {FREQ}Hz")
    print(f"数据将实时保存至当前目录下的 {csv_filename} 文件。\n")

    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            previous_data = None
            
            while True:
                start_time = time.time()
                
                try:
                    response = requests.get(URL, timeout=0.04)
                    
                    if response.status_code == 200:
                        raw_data = response.json()
                        
                        if "TAS, km/h" in raw_data:
                            current_data = [raw_data.get(key, 0) for key in KEYS_TO_EXTRACT]
                            
                            # 数据查重
                            if current_data != previous_data:
                                current_timestamp = time.time()
                                row = [current_timestamp] + current_data
                                writer.writerow(row)
                                f.flush()
                                previous_data = current_data
                                
                                # 控制台预览
                                h = raw_data.get("H, m", 0)
                                tas = raw_data.get("TAS, km/h", 0)
                                g = raw_data.get("Ny", 1)
                                print(f"\r[{time.strftime('%H:%M:%S')}] 记录中 | 高度: {h:4}m | 空速: {tas:4}km/h | 过载: {g:4.1f}G", end="")
                        else:
                            print(f"\r未检测到有效飞行状态。{' '*20}", end="")
                
                except requests.exceptions.RequestException:
                    print(f"\r无法连接到 8111 端口，正在等待游戏运行...{' '*20}", end="")
                except ValueError:
                    pass

                # 动态休眠补偿
                elapsed = time.time() - start_time
                sleep_time = max(0, INTERVAL - elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        # Ctrl+C 结束采集
        print(f"\n\n采集程序已安全退出，数据已保存至 {csv_filename} 。")

if __name__ == "__main__":
    main()