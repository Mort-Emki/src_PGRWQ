import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from data_processing import load_daily_data, load_river_attributes
from model_training.train import iterative_training_procedure
import os

# 设置数据根目录
os.chdir(r"D:\\PGRWQ\\data")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--model_type", type=str, default="lstm", help="'rf' 或 'lstm'")
    # 新增：指定输入特征列表（逗号分隔），例如 "Feature1,Feature2,Feature3,Feature4,Feature5"
    parser.add_argument("--input_features", type=str, default="Feature1,Feature2,Feature3,Feature4,Feature5",
                        help="以逗号分隔的输入特征名称列表")
    # 新增：指定属性特征列表（逗号分隔），例如 "Attr1,Attr2,Attr3"
    parser.add_argument("--attr_features", type=str, default="Attr1,Attr2,Attr3",
                        help="以逗号分隔的属性特征名称列表")
    args = parser.parse_args()

    # 将逗号分隔的字符串转换为列表
    input_features = [feat.strip() for feat in args.input_features.split(",") if feat.strip()]
    attr_features = [feat.strip() for feat in args.attr_features.split(",") if feat.strip()]


    ## 手动指定输入特征列表和属性特征列表
    input_features = ['surface_net_solar_radiation_mean', 'surface_pressure_mean', 'temperature_2m_mean', 'u_component_of_wind_10m_mean', 'v_component_of_wind_10m_mean',
        'volumetric_soil_water_layer_1_mean','volumetric_soil_water_layer_2_mean','temperature_2m_min','temperature_2m_max','total_precipitation_sum','potential_evaporation_sum','Qout']

    attr_features =  [
    # 气候因子
    'pre_mm_syr',         # 年降水量，反映区域总体降水输入
    'pet_mean',           # 日均潜在蒸散，代表蒸发能力
    'aridity',            # 干旱指数，体现水分供需平衡
    'seasonality',        # 湿润指数的季节性，反映降水分布的不均性
    'high_prec_freq',     # 极端降水事件的频率，对面源污染具有脉冲效应

    # 土地利用
    'crp_pc_sse',         # 耕地比例，代表农业活动及化肥使用
    'for_pc_sse',         # 森林覆盖率，有助于截留和吸收营养盐
    'urb_pc_sse',         # 城市用地比例，反映城市化与生活污水排放
    'wet_pc_s01',         # 湿地比例，湿地具有调节营养盐的功能

    # 人类活动
    'nli_ix_sav',         # 夜间灯光指标，遥感反映人类活动密度
    'pop_ct_usu',         # 汇流口人口总数，直接关联生活污染

    # 水文因子
    'dis_m3_pyr',         # 年河流流量，决定稀释与传输过程
    'run_mm_syr',         # 地表径流深，指示面源污染传递能力

    # 土壤因子
    'cly_pc_sav',         # 土壤黏土比例，影响养分吸附与保留
    'soc_th_sav',         # 土壤有机碳含量，反映土壤微生物作用及营养盐固定

    # 地形因子
    'ele_mt_sav',         # 平均高程，表征区域地势特征
    'slp_dg_sav',         # 坡度，影响侵蚀和径流强度
    'sgr_dk_sav',         # 河道坡降，直接关联水流动力及输运效率

    # 补充因子
    'moisture_index',     # 湿润指数，综合反映区域水分状况
    'ero_kh_sav'          # 土壤侵蚀速率，影响沉积物及磷的输送
    ]

    # 根据指定列表自动调整 input_dim 和 attr_dim
    input_dim = len(input_features)
    attr_dim = len(attr_features)
    print(f"输入特征列表: {input_features} (维度: {input_dim})")
    print(f"属性特征列表: {attr_features} (维度: {attr_dim})")

    # 检查 GPU 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # 加载日尺度数据（feature_daily_ts.csv），文件中需包含 'COMID'、'Date'、驱动特征、'Qout'、'TN'、'TP' 等字段
    daily_csv = "feature_daily_ts.csv"
    df = load_daily_data(daily_csv)
    print("日尺度数据基本信息：")
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}")
    # print("前5条记录：")
    # print(df.head())

    # 加载河段属性数据（river_attributes.csv），文件中需包含 'COMID'、'NextDownID' 以及其他属性
    attr_df = load_river_attributes("river_attributes_new.csv")
    print("\n河段属性数据基本信息：")
    print(f"  数据形状: {attr_df.shape}")
    print(f"  列名: {attr_df.columns.tolist()}")
    # print("前5条记录：")
    # print(attr_df.head())

    # 将河段属性数据转换为字典，键为字符串形式的 COMID，
    # 值为一个数组，仅包含用户指定的属性特征（如果某属性不存在则置 0）
    attr_dict = {}
    for row in attr_df.itertuples(index=False):
        comid = str(row.COMID)
        row_dict = row._asdict()
        attrs = []
        for attr in attr_features:
            if attr in row_dict:
                attrs.append(row_dict[attr])
            else:
                attrs.append(0.0)
        attr_dict[comid] = np.array(attrs, dtype=np.float32)


    # print(attr_dict)


    # 提取河段信息，这里只需 COMID 和 NextDownID（NextDownID 来自属性数据）
    river_info = attr_df[['COMID', 'NextDownID']].copy()
    # 确保 NextDownID 为数字；若存在缺失值则填为 0
    river_info['NextDownID'] = pd.to_numeric(river_info['NextDownID'], errors='coerce').fillna(0).astype(int)
    
    ##        comid_wq_list and comid_era5_list，分别从WQ_exist_comid.csv和ERA5_exist_comid.csv中提取出存在的COMID列表
    comid_wq_list = pd.read_csv("WQ_exist_comid.csv", header=None)[0].tolist()
    comid_era5_list = pd.read_csv("ERA5_exist_comid.csv", header=None)[0].tolist()


    # 调用迭代训练过程，传入 input_cols 参数为 input_features
    final_model = iterative_training_procedure(
        df=df,
        attr_df = attr_df,
        # attr_dict=attr_dict,
        input_features   = input_features,
        attr_features    = attr_features,
        river_info=river_info,
        target_cols=['TN','TP'],
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        model_type=args.model_type,
        input_dim=input_dim,
        hidden_size=64,
        num_layers=1,
        attr_dim=attr_dim,
        fc_dim=32,
        device=device,
        comid_wq_list=comid_wq_list,
        comid_era5_list=comid_era5_list,
        input_cols=input_features
    )
    
    print("迭代训练完成，最终模型已训练完毕。")

if __name__ == "__main__":
    main()
