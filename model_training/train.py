import numpy as np
import pandas as pd
from typing import List
from data_processing import build_sliding_windows_for_subset, standardize_time_series_all, standardize_attributes
from model_training.models import CatchmentModel
from flow_routing import flow_routing_calculation
from tqdm import tqdm
import numba
import time
import os


def iterative_training_procedure(df: pd.DataFrame,
                                 attr_df: pd.DataFrame,
                                #  attr_dict: dict,
                                 input_features: List[str] = None,
                                 attr_features: List[str] = None,
                                 river_info: pd.DataFrame = None,
                                 target_cols: List[str] = ["TN","TP"],
                                 target_col: str = "TN",
                                 max_iterations: int = 10,
                                 epsilon: float = 0.01,
                                 model_type: str = 'rf',
                                 input_dim: int = None,
                                 hidden_size: int = 64,
                                 num_layers: int = 1,
                                 attr_dim: int = None,
                                 fc_dim: int = 32,
                                 device: str = 'cpu',
                                 comid_wq_list: list = None,
                                 comid_era5_list: list = None,
                                 input_cols: list = None):
    """
    迭代训练过程
    输入：
        df: 日尺度数据 DataFrame，包含 'COMID'、'Date'、target_col、'Qout' 等字段
        attr_dict: 河段属性字典，键为 str(COMID)，值为属性数组（已标准化）
        river_info: 河段信息 DataFrame，包含 'COMID' 和 'NextDownID'
        target_col: 目标变量名称，如 "TP"
        max_iterations: 最大迭代次数
        epsilon: 收敛阈值（残差最大值）
        model_type: 'rf' 或 'lstm'
        input_dim: 模型输入维度（须与 input_cols 长度一致）
        hidden_size, num_layers, attr_dim, fc_dim: 模型参数
        device: 训练设备
        input_cols: 指定用于构造时间序列输入的特征列表（例如，["Feature1", "Feature2", ...]）
    输出：
        返回训练好的模型对象
    过程：
        1. 选择头部河段（未出现在 river_info 中 NextDownID 不为 0 的 COMID）进行初始模型训练（A₀）
           （局部贡献 E 直接以观测值作为目标）。
        2. 利用 A₀ 对全数据进行汇流计算，得到各河段的 y_n 与 y_up。
        3. 计算残差（观测值 - y_n），若最大残差小于 epsilon 则收敛，否则更新标签：E_label = 观测值 - y_up。
        4. 利用滑窗数据切片及真实属性数据构造训练数据，重新训练模型，并更新汇流结果。
    """
    print('选择头部河段进行初始模型训练。')
    # # 选择头部河段：从 river_info 中提取 NextDownID 信息
    # downstream_ids = river_info['NextDownID'][river_info['NextDownID'] != 0].unique()
    # head_mask = ~df['COMID'].isin(downstream_ids)
    # df_head = df[head_mask].copy()
    # df_head[target_col] = df_head[target_col].fillna(0.0)

    ##attr_df中 order_<=2 的河段，为head_upstream, 建立df_head_upstream
    attr_df_head_upstream = attr_df[attr_df['order_'] <= 2]
    # attr_df_head_upstream = attr_df
    df_head_upstream = df[df['COMID'].isin(attr_df_head_upstream['COMID'])]
    # df_head_upstream[target_col] = df_head_upstream[target_col].fillna(0.0)

    # ##统计df_head_upstream target_col的缺值率
    # missing_rate = df_head_upstream[target_col].isna().sum() / len(df_head_upstream)
    # print(f"  初始训练集缺失率：{missing_rate:.2%}")

    # #打印查看COMID为43050338的缺值情况
    # print(df_head_upstream[df_head_upstream['COMID']==43050338][target_col].isna().sum() / len(df_head_upstream[df_head_upstream['COMID']==43050338]))
    
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


    ##comid_list_head取df_head中的comid与comid_wq_list、comid_era5_list的交集，当comid_wq_list、comid_era5_list为空时，打印错误信息并退出程序
    if comid_wq_list is None:
        comid_wq_list = []
    if comid_era5_list is None:
        comid_era5_list = []
    comid_list_head = list(set(df_head_upstream['COMID'].unique().tolist()) & set(comid_wq_list) & set(comid_era5_list))
    if len(comid_list_head) == 0:
        print("警告：comid_wq_list、comid_era5_list 为空，请检查输入。")
        return None
    print(f"  选择的头部河段数量：{ len(comid_list_head)}")

    ##统计df_head_upstream中cmoid等于comid_list_head中的 target_col的缺值率
    # missing_rate = df_head_upstream[df_head_upstream['COMID'].isin(comid_list_head)][target_col].isna().sum() / len(df_head_upstream[df_head_upstream['COMID'].isin(comid_list_head)])
    # print(f"  初始训练集缺失率：{missing_rate:.2%}")

    print('构造初始训练数据（滑窗切片）......')
    # print(comid_list_head)
    ##保存comid_list_head到txt文件
    # with open("comid_list_head.txt", "w") as f:
    #     for comid in comid_list_head:
    #         f.write(str(comid) + "\n")
    # print("comid_list_head保存成功！")

    # 构造初始训练数据（滑窗切片），传入 input_cols 参数
    X_ts_head, Y_head_orig, COMIDs_head, Dates_head = build_sliding_windows_for_subset(
        df, 
        comid_list_head, 
        input_cols=None, 
        target_cols=target_cols, 
        time_window=10
    )
    Y_head = Y_head_orig[:,0]

    print("X_ts_all.shape =", X_ts_head.shape)
    print("Y.shape        =", Y_head.shape)
    print("COMID.shape    =", COMIDs_head.shape)  
    print("Date.shape     =", Dates_head.shape)

    ##保存到npz文件
    np.savez("upstreams_trainval_mainsrc.npz", X=X_ts_head, Y=Y_head_orig, COMID=COMIDs_head, Date=Dates_head)
    print("训练数据保存成功！")


################

    # import os
    # os.chdir("D:\\PGRWQ\\data")  # Set root directory
    # # loaded = np.load("trainval_allstations_includingTest80.npz", allow_pickle=True)
    # loaded = np.load("upstreams_trainval.npz", allow_pickle=True)
    # X_ts_all = loaded["X"]        # (N, T, input_dim)
    # Y_all = loaded["Y"]           # (N,) or (N, something)
    # Y = Y_all[:, 0]               # We take the 0th column (TP), shape=(N,)
    # comid_array_all = loaded["COMID"]  # (N,)
    # date_array_all  = loaded["Date"]   # (N,)
    
    # print("Data loaded (for training+validation):")
    # print("X_ts_all.shape =", X_ts_all.shape)
    # print("Y.shape        =", Y.shape)
    # print("COMID.shape    =", comid_array_all.shape)  
    # print("Date.shape     =", date_array_all.shape)

    # # # %% [3] Load river attributes
    # # attr_df = pd.read_csv("river_attributes.csv")  # columns=[COMID, p_mean, aridity, ...]
    # # attr_dict = {}
    # # for row in attr_df.itertuples(index=False):
    # #     comid = str(row.COMID)
    # #     attrs = [v for k, v in row._asdict().items() if k != "COMID"]
    # #     attr_dict[comid] = np.array(attrs, dtype=np.float32)

    # # Check for NaNs
    # print("Any X_ts NaN?", np.isnan(X_ts_all).any())
    # print("Any Y NaN?", np.isnan(Y).any())
    # print("X_ts NaN count:", np.count_nonzero(np.isnan(X_ts_all)))

    # X_ts_head = X_ts_all
    # COMIDs_head = comid_array_all
    # Y_head = Y_all[:, 0]

################


    X_ts_head_scaled, ts_scaler = standardize_time_series_all(X_ts_head)
    attr_dict_scaled, attr_scaler = standardize_attributes(attr_dict)

    X_ts_head = X_ts_head_scaled
    attr_dict = attr_dict_scaled

    N, T, input_dim = X_ts_head.shape
    attr_dim = len(next(iter(attr_dict.values())))


    ##划分train和validation
    N = len(X_ts_head)
    indices = np.random.permutation(N)
    train_size = int(N * 0.8)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]


    X_ts_train = X_ts_head[train_indices]
    comid_arr_train = COMIDs_head[train_indices]
    Y_train = Y_head[train_indices]

    X_ts_val = X_ts_head[valid_indices]
    comid_arr_val = COMIDs_head[valid_indices]
    Y_val = Y_head[valid_indices]

    # print(X_ts_train)
    # print(X_ts_val)

    print("初始模型 A₀ 训练：头部河段训练数据构造完毕。")
    
    model = CatchmentModel(model_type=model_type,
                           input_dim=input_dim,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           attr_dim=attr_dim,
                           fc_dim=fc_dim,
                           device=device)

    ##如果指定路径的model不存在，则训练模型，否则直接加载模型
    model_path = "model_initial_A0.pth"
    if not os.path.exists(model_path):
        model.train_model(attr_dict, comid_arr_train, X_ts_train, Y_train, comid_arr_val, X_ts_val, Y_val, epochs=100, lr=1e-3, patience=2, batch_size=32)
        model.save_model(model_path)
        print("模型训练成功！")
    else:
        model.load_model(model_path)
        print("模型加载成功！")

    # model.train_model(attr_dict, comid_arr_train, X_ts_train, Y_train, comid_arr_val, X_ts_val, Y_val, epochs=100, lr=1e-3, patience=2, batch_size=32)
    
    # ###保存模型
    # model.save_model("model_initial_A0.pth")
    # print("模型保存成功！")
    
    def initial_model_func(group: pd.DataFrame, attr_dict: dict, model: CatchmentModel):
        group_sorted = group.sort_values("date")
        X_ts_local, _, _, Dates_local = build_sliding_windows_for_subset(
            df=group, 
            comid_list=[group.iloc[0]['COMID']], 
            input_cols=None, 
            target_cols=target_cols, 
            time_window=10,
            skip_missing_targets=False
        )
        
        if X_ts_local is None:
            print(f"警告：COMID {group.iloc[0]['COMID']} 数据不足，返回 0。")
            return pd.Series(0.0, index=group_sorted["date"])
        
        comid_str = str(group.iloc[0]['COMID'])
        attr_vec = attr_dict.get(comid_str, np.zeros_like(next(iter(attr_dict.values()))))
        X_attr_local = np.tile(attr_vec, (X_ts_local.shape[0], 1))
        preds = model.predict(X_ts_local, X_attr_local)
        
        # 创建预测序列，使用实际预测日期做索引
        pred_series = pd.Series(preds, index=pd.to_datetime(Dates_local))
        
        # 创建完整序列，使用原始数据的所有日期做索引，默认值为0
        full_series = pd.Series(0.0, index=group_sorted["date"])
        
        # 用预测值更新完整序列中对应的日期
        full_series.update(pred_series)
        
        return full_series   
    
    

    print("初始汇流计算：使用 A₀ 进行预测。")
    df_flow = flow_routing_calculation(df = df.copy(), 
                                       iteration=0, 
                                       model_func=initial_model_func, 
                                       river_info=river_info, 
                                       v_f=35.0,
                                       attr_dict=attr_dict,
                                       model=model)
    
    # 迭代更新过程
    for it in range(max_iterations):
        print(f"\n迭代 {it+1}/{max_iterations}")
        col_y_n = f'y_n_{it}'    
        col_y_up = f'y_up_{it}'
        merged = pd.merge(df, df_flow[['COMID', 'Date', col_y_n, col_y_up]], on=['COMID', 'Date'], how='left')
        y_true = merged[target_col].values
        y_pred = merged[col_y_n].values
        residual = y_true - y_pred
        max_resid = np.abs(residual).max()
        print(f"  最大残差: {max_resid:.4f}")
        if max_resid < epsilon:
            print("收敛！")
            break
        merged["E_label"] = merged[target_col] - merged[col_y_up]
        comid_list_iter = merged["COMID"].unique().tolist()
        X_ts_iter, _, COMIDs_iter, Dates_iter = build_sliding_windows_for_subset(
            df, comid_list_iter, input_cols=input_cols, target_cols=[target_col], time_window=5
        )
        Y_label_iter = []
        for cid, date_val in zip(COMIDs_iter, Dates_iter):
            subset = merged[(merged["COMID"] == cid) & (merged["Date"] == date_val)]
            if not subset.empty:
                label_val = subset["E_label"].mean()
            else:
                label_val = 0.0
            Y_label_iter.append(label_val)
        Y_label_iter = np.array(Y_label_iter, dtype=np.float32)
        X_attr_iter = np.vstack([attr_dict.get(str(cid), np.zeros_like(next(iter(attr_dict.values()))))
                                  for cid in COMIDs_iter])
        print("  更新模型训练：使用更新后的 E_label。")
        model.train_model(X_ts_iter, X_attr_iter, Y_label_iter, epochs=5, lr=1e-3, patience=2, batch_size=32)
        
        # 更新后的模型函数：利用真实切片数据和属性数据进行预测
        def updated_model_func(group: pd.DataFrame):
            group_sorted = group.sort_values("Date")
            X_ts_local, _, _, Dates_local = build_sliding_windows_for_subset(
                group, [group.iloc[0]['COMID']], input_cols=input_cols, target_cols=[target_col], time_window=5
            )
            if X_ts_local is None:
                print(f"警告：COMID {group.iloc[0]['COMID']} 数据不足，返回 0。")
                return pd.Series(0.0, index=group_sorted["Date"])
            comid_str = str(group.iloc[0]['COMID'])
            attr_vec = attr_dict.get(comid_str, np.zeros_like(next(iter(attr_dict.values()))))
            X_attr_local = np.tile(attr_vec, (X_ts_local.shape[0], 1))
            preds = model.predict(X_ts_local, X_attr_local)
            return pd.Series(preds, index=pd.to_datetime(Dates_local))
        
        df_flow = flow_routing_calculation(df.copy(), iteration=it+1, model_func=updated_model_func, river_info=river_info, v_f=35.0)
    
    return model
