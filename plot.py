import torch
import matplotlib.pyplot as plt

def plotResults(params, trials, device):
    with torch.no_grad():
        x_plot = torch.linspace(0.0, 1.0, 100, device=device).unsqueeze(1)
        u_plot = trials.u_trial(x_plot).detach().cpu().numpy().squeeze()
        x_plot_np = x_plot.detach().cpu().numpy().squeeze()
        x_data_np = params.x_data.detach().cpu().numpy().squeeze()
        u_data_np = params.u_data.detach().cpu().numpy().squeeze()
        
        fig, ax = plt.subplots()
        ax.plot(x_plot_np, u_plot, 'blue')
        ax.plot(x_data_np, u_data_np, 'ko', label='Data Points')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.grid(True)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)
