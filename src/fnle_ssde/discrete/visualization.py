from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


def calc_log_prob_mesh(nle, state, hmm_param_true, X_obs_concat, 
                       targets, theta_idx_as_x, theta_idx_as_y):
    # p(X=X^* | theta)
    tmp_theta = hmm_param_true['emissions'][state].expand(
        X_obs_concat.shape[0], -1).clone()
    # Generate dummy data
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = []
    for xi, yi in tqdm(zip(X, Y)):
        zi = []
        for xij, yij in zip(xi, yi):
            tmp_theta[:, theta_idx_as_x] = xij
            tmp_theta[:, theta_idx_as_y] = yij
            conditions = torch.cat(
                [tmp_theta, X_obs_concat], dim=-1).to(torch.float32)
            log_prob = nle.estimator.log_prob(
                targets.unsqueeze(0), condition=conditions).sum().item()
            zi.append(log_prob)
        Z.append(zi)
    return X, Y, Z


def plot_log_prob_contour(nle, state, hmm_param_true, X_obs_concat, targets,
                          theta_idx_as_x, theta_idx_as_y):
    X, Y, Z = calc_log_prob_mesh(nle, state, hmm_param_true, X_obs_concat,
                                 targets, theta_idx_as_x, theta_idx_as_y)
    cs = plt.contour(X, Y, Z, levels=20)
    cbar = plt.colorbar(cs)
    plt.scatter(hmm_param_true['emissions'][state][theta_idx_as_x],
                hmm_param_true['emissions'][state][theta_idx_as_y],
                marker='x', color='red', s=100,
                label="True parameters", zorder=3)
    plt.xlabel('Log (gamma)')
    plt.ylabel('Log (delta)')
    plt.legend()
    plt.show()


def plot_series_multi(
    t_list, X_list, X_obs_list, Z_list, Z_est_list, obs_idx_list,
    var_names=None, figsize=(10, 3.0), panel_h=2.9,
):
    """
    X_list:     [X_i], 各 X_i は shape (2, T)
    Z_list:     [Z_i], 各 Z_i は shape (T,)
    Z_est_list: [Zest_i], 各 Zest_i は shape (T,)
    """
    n = len(X_list)
    if var_names is None: var_names = [f'X[{i}]' for i in range(X_list[0].shape[0])]

    total_rows = 2 * n
    fig_h = max(panel_h * n, figsize[1])
    fig, axes = plt.subplots(total_rows, 1, sharex=True, figsize=(figsize[0], fig_h))
    axes = np.atleast_1d(axes)

    # 柔らかい色
    color_Z      = "#3182bd"
    color_Zest   = "#e6550d"
    color_X0     = "#8040fc"
    color_X1     = "#4bae06"

    handles_all, labels_all = [], []

    for i, (Z, obs_idx, z_est, X, x_obs) in enumerate(
            zip(Z_list, obs_idx_list, Z_est_list, X_list, X_obs_list)):
        series_name = f"Series{i+1}"

        # 軸
        axZ = axes[2*i]
        axX = axes[2*i + 1]

        # --- 上段: Z & Z_est
        h1 = axZ.plot(t_list, Z, drawstyle="steps-post",
                      label="Ground truth", color=color_Z)[0]
        h2 = axZ.scatter(t_list[obs_idx][1:], z_est,
                         label="Estimated", color=color_Zest, s=15, alpha=0.8)
        axZ.set_ylim(-0.1, 1.1)
        axZ.set_ylabel("Hidden states")
        axZ.grid(True, alpha=0.3)
        # タイトルにシリーズ名
        axZ.set_title(series_name, loc="left", fontsize=11, fontweight="bold")

        # --- 下段: X & X_obs
        h3 = axX.plot(t_list, X[0], label=var_names[0], color=color_X0)[0]
        h4 = axX.plot(t_list, X[1], label=var_names[1], color=color_X1)[0]
        axX.scatter(t_list[obs_idx], x_obs[0], s=28, label="X_obs[0]",
                    marker='o', zorder=3, edgecolors="none", color=color_X0)
        axX.scatter(t_list[obs_idx], x_obs[1], s=28, label="X_obs[1]",
                    marker='x', zorder=3, color=color_X1)
        axX.set_ylabel("Observations")
        axX.grid(True, alpha=0.3)

        if i == n-1:
            axX.set_xlabel("Time")
        else:
            axX.tick_params(labelbottom=False)

        # 凡例用ハンドル収集（重複除外）
        for h, lb in [(h1, "Ground truth"), (h2, "Estimated"),
                      (h3, var_names[0]), (h4, var_names[1])]:
            if lb not in labels_all:
                labels_all.append(lb)
                handles_all.append(h)

    # 図全体の凡例（上にまとめる）
    fig.legend(handles_all, labels_all, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.98), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
