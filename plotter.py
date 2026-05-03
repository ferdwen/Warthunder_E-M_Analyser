import tkinter as tk
from tkinter import messagebox

def run_analysis(aircraft_model, target_alt, alt_tol, max_vy, max_roll_rate, max_sideslip, throttle, sigma_val, use_hybrid):
    import os
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    
    # ==========================================
    # 1. 硬编码常量
    # ==========================================
    max_dt = 0.125 # dt 最大值限制 (s)
    max_sep_noise = 1000.0 # SEP 最大绝对值限制 (m/s)

    # ==========================================
    # 2. 数据读取
    # ==========================================
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("当前目录下没有找到任何 CSV 文件，请确认文件位置。")
        return
        
    print(f"找到 {len(csv_files)} 份试飞文件，正在读取数据。")
    
    df_list = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file, engine='pyarrow')
            df_list.append(temp_df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            
    # 融合成大表，按绝对时间排序
    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=['timestamp'], inplace=True, ignore_index=True)
    
    # ==========================================
    # 3. 微积分
    # ==========================================
    g = 9.81
    
    # 全局差分计算
    df['dt'] = df['timestamp'].diff()
    df['V_ms'] = df['TAS, km/h'] / 3.6
    df['dV'] = df['V_ms'].diff()
    
    # 侦测物理断层并插入 NaN
    boundary_mask = df['dt'].isna() | (df['dt'] > max_dt)
    df.loc[boundary_mask, ['dt', 'dV']] = np.nan

    # 计算加速度
    df['a_raw'] = df['dV'] / df['dt']
    
    # 滑动平均，NaN 自动触发物理隔离
    df['a_smooth'] = df['a_raw'].rolling(window=7, min_periods=7).mean()

    # 计算转向率和 SEP
    ny_safe = np.maximum(df['Ny'], 1.0)
    df['turn_rate'] = np.degrees((g * np.sqrt(ny_safe**2 - 1)) / df['V_ms'])
    df['SEP'] = df['Vy, m/s'] + (df['V_ms'] / g) * df['a_smooth']

    # ==========================================
    # 4. 数据清洗
    # ==========================================
    original_len = len(df)
    
    alt_lower = target_alt - alt_tol
    alt_upper = target_alt + alt_tol
    
    # 数据清洗条件
    query_str = (
        f"{alt_lower} <= `H, m` <= {alt_upper} and "
        f"`throttle 1, %` == {throttle} and "
        f"abs(`Vy, m/s`) <= {max_vy} and "
        f"abs(`Wx, deg/s`) <= {max_roll_rate} and "
        f"abs(`AoS, deg`) <= {max_sideslip} and "
        "`gear, %` == 0 and `airbrake, %` == 0 and `flaps, %` == 0 and "
        f"abs(SEP) < {max_sep_noise}"
    )
    
    # 执行清洗并拷贝出干净的底表
    df_clean = df.query(query_str).copy()
    df_clean.dropna(subset=['V_ms', 'turn_rate', 'SEP'], inplace=True)
    
    valid_len = len(df_clean)
    print(f"数据清洗完成。原始数据: {original_len} 行 > 有效数据: {valid_len} 行。")
    
    if valid_len < 10:
        print("警告：有效数据过少（不足 10 条）。")
        print(f"请确保在游戏中以目标油门({throttle}%)、{target_alt}米高度、收起所有襟翼/减速板/起落架进行纯水平盘旋测试。")
        return

    # ==========================================
    # 5. 空间网格预处理
    # ==========================================
    # 利用速度为纯整数的特性，在坐标轴上对高度重叠点进行合并求均值
    df_clean['X_round'] = df_clean['TAS, km/h']
    df_clean['Y_round'] = df_clean['turn_rate'].round(1)
    
    df_reduced = df_clean.groupby(['X_round', 'Y_round'], as_index=False)['SEP'].mean()
    
    # 精简后的维度数据交由插值器处理
    x_data = df_reduced['X_round'].values
    y_data = df_reduced['Y_round'].values
    z_data = df_reduced['SEP'].values

    # ==========================================
    # 6. 混合空间插值与渲染成图
    # ==========================================
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8), dpi=180)

    # 对称色带标尺
    max_abs_sep = df_clean['SEP'].abs().max()
    color_limit = np.ceil(max_abs_sep * 0.75) 

    # 绘制基础散点
    scatter = plt.scatter(
        df_clean['TAS, km/h'], df_clean['turn_rate'],
        c=df_clean['SEP'], cmap='RdYlGn',
        vmin=-color_limit, vmax=color_limit, 
        alpha=0.6, edgecolors='none', s=10,
        rasterized=True  
    )

    # 生成致密网格 (防止插值断层)
    grid_x, grid_y = np.mgrid[
        max(0, x_data.min() - 50) : x_data.max() * 1.05 : 1920j, 
        0 : y_data.max() * 1.05 : 1080j
    ]

    # 基础线性插值
    grid_z_linear = griddata((x_data, y_data), z_data, (grid_x, grid_y), method='linear')

    # 根据 UI 开关决定是否启用邻近插值填补空洞
    if use_hybrid:
        grid_z_nearest = griddata((x_data, y_data), z_data, (grid_x, grid_y), method='nearest')
        grid_z_raw = np.where(np.isnan(grid_z_linear), grid_z_nearest, grid_z_linear)
    else:
        grid_z_raw = grid_z_linear.copy()

    # NaN 感知的高斯模糊
    V = grid_z_raw.copy()
    V[np.isnan(V)] = 0
    W = np.ones_like(grid_z_raw)
    W[np.isnan(grid_z_raw)] = 0
    
    V_smooth = gaussian_filter(V, sigma=sigma_val)
    W_smooth = gaussian_filter(W, sigma=sigma_val)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        grid_z_smooth = V_smooth / W_smooth

    # 物理遮罩，修剪掉距离真实数据太远的虚空等高线
    x_range = x_data.max() - x_data.min() if x_data.max() != x_data.min() else 1
    y_range = y_data.max() - y_data.min() if y_data.max() != y_data.min() else 1
    
    x_norm, y_norm = x_data / x_range, y_data / y_range
    grid_x_norm, grid_y_norm = grid_x / x_range, grid_y / y_range
    
    tree = cKDTree(np.c_[x_norm, y_norm])
    dist, _ = tree.query(np.c_[grid_x_norm.ravel(), grid_y_norm.ravel()])
    dist = dist.reshape(grid_x.shape)
    
    # 距离阈值，超出这个范围的线将被遮蔽
    mask_threshold = 0.07
    grid_z_smooth[dist > mask_threshold] = np.nan

    # ----------------------------------------------------
    # 7. 视觉元素
    # ----------------------------------------------------
    # 等高线位置与修饰
    target_levels = [-300, -200, -100, 0, 100, 200, 300]
    line_colors = ['white', 'white', 'white', 'red', 'white', 'white', 'white']
    line_widths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    contour = plt.contour(
        grid_x, grid_y, grid_z_smooth, 
        levels=target_levels, colors=line_colors, linewidths=line_widths, zorder=5
    )
    
    # 图表修饰
    plt.clabel(contour, inline=True, fontsize=8, fmt='SEP = %d')

    cbar = plt.colorbar(scatter)
    cbar.set_label('SEP [m/s]', fontsize=12, fontweight='bold')
    
    plt.title(f'{aircraft_model} Energy-Maneuverability Diagram\n[Alt: {target_alt}m ± {alt_tol}m | Throttle: {throttle}%]', 
              fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('TAS [km/h]', fontsize=12, fontweight='bold')
    plt.ylabel('Turn Rate [deg/s]', fontsize=12, fontweight='bold')
    
    plt.axhline(y=0, color='white', linestyle='-', alpha=1.0)
    plt.grid(True, linestyle=':', alpha=0.75)
    plt.ylim(bottom=0)
    plt.xlim(left=max(0, x_data.min() - 100))

    plt.tight_layout()
    plt.show()

# ==========================================
# 8. 交互界面
# ==========================================
def create_gui():
    root = tk.Tk()
    root.title("E-M 包线分析配置")
    
    # 窗口居中
    window_width = 280
    window_height = 350
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width / 2) - (window_width / 2))
    y = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    entries = {}
    
    def create_input(row, label_text, default_value=""):
        tk.Label(root, text=label_text).grid(row=row, column=0, padx=20, pady=5, sticky='w')
        entry = tk.Entry(root, width=15)
        if default_value != "":
            entry.insert(0, str(default_value))
        entry.grid(row=row, column=1, padx=5, pady=5)
        return entry

    # 创建输入框并赋予默认值
    entries['model']    = create_input(0, "飞机机型:")
    entries['alt']      = create_input(1, "期望测试高度 (m):", 50.0)
    entries['alt_tol']  = create_input(2, "高度容差 (m):", 50.0)
    entries['vy']       = create_input(3, "爬升率容差 (m/s):", 20.0)
    entries['roll']     = create_input(4, "滚转率容差 (deg/s):", 90.0)
    entries['slip']     = create_input(5, "侧滑角容差 (deg):", 12.5)
    entries['throttle'] = create_input(6, "测试油门大小 (%):", 110.0)
    entries['sigma']    = create_input(7, "西格玛值:", 64.0)
    use_hybrid_var = tk.BooleanVar(value=True)
    chk_hybrid = tk.Checkbutton(root, text="启用混合插值法", variable=use_hybrid_var)
    chk_hybrid.grid(row=8, column=0, columnspan=2, pady=5)

    def on_submit():
        try:
            # 读取数据
            val_model = entries['model'].get().strip()
            val_alt   = float(entries['alt'].get())
            val_tol   = float(entries['alt_tol'].get())
            val_vy    = float(entries['vy'].get())
            val_roll  = float(entries['roll'].get())
            val_slip  = float(entries['slip'].get())
            val_thr   = float(entries['throttle'].get())
            val_sigma = float(entries['sigma'].get())
            val_hybrid = use_hybrid_var.get()
        except ValueError:
            messagebox.showerror("无效字符", "请确保除机型外所有输入框内都填写了有效的数字。")
            return
            
        # 隐藏输入窗口
        root.withdraw()
        
        # 传入所有参数执行画图
        run_analysis(val_model, val_alt, val_tol, val_vy, val_roll, val_slip, val_thr, val_sigma, val_hybrid)
        
        # 执行完毕后销毁
        root.destroy()

    submit_btn = tk.Button(root, text="生成图表", font=('', 10, 'bold'), command=on_submit)
    submit_btn.grid(row=9, column=0, columnspan=2, pady=5, ipadx=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()