import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "hiv_stage1")

def plot_lambda():
    lambda_path = os.path.join(RESULTS_DIR, "hiv_lambda.npy")
    if not os.path.exists(lambda_path):
        print(f"File not found: {lambda_path}")
        return
    lambdas = np.load(lambda_path)
    plt.figure(figsize=(10,4))
    plt.plot(lambdas, marker='o')
    plt.xlabel("Position")
    plt.ylabel("Mutation rate (λ)")
    plt.title("Per-site mutation rates for HIV")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hiv_lambda_plot.png"))
    plt.show()

def plot_dca_couplings():
    J_path = os.path.join(RESULTS_DIR, "hiv_J.npz")
    if not os.path.exists(J_path):
        print(f"File not found: {J_path}")
        return
    J = np.load(J_path)["J"]
    J_frob = np.sqrt((J ** 2).sum(axis=(2, 3)))
    plt.figure(figsize=(8,6))
    plt.imshow(J_frob, cmap="viridis")
    plt.colorbar(label="||J[i,j]||_F")
    plt.title("DCA Coupling Strengths (HIV)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hiv_J_frob_plot.png"))
    plt.show()

if __name__ == "__main__":
    plot_lambda()
    plot_dca_couplings()

    # Additional heatmaps for further evaluation
    import seaborn as sns

    # Plot heatmap for h (fields)
    h_path = os.path.join(RESULTS_DIR, "hiv_h.npy")
    if os.path.exists(h_path):
        h = np.load(h_path)
        plt.figure(figsize=(12, 6))
        sns.heatmap(h, cmap="coolwarm", cbar=True)
        plt.title("DCA Field Parameters h (positions × 20 AAs)")
        plt.xlabel("Amino Acid Index")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "hiv_h_heatmap.png"))
        plt.show()
    else:
        print(f"File not found: {h_path}")

    # Plot heatmap for J (couplings) for a selected AA pair (e.g., A=0, A=0)
    J_path = os.path.join(RESULTS_DIR, "hiv_J.npz")
    if os.path.exists(J_path):
        J = np.load(J_path)["J"]
        # Show J[:,:,0,0] as an example (coupling between AA 0 at all positions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(J[:, :, 0, 0], cmap="coolwarm", cbar=True)
        plt.title("DCA Coupling J[:,:,0,0] (Position × Position, AA=0)")
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "hiv_J_aa0_heatmap.png"))
        plt.show()
        # Optionally, plot the mean coupling over all AA pairs
        plt.figure(figsize=(10, 8))
        sns.heatmap(J.mean(axis=(2,3)), cmap="coolwarm", cbar=True)
        plt.title("Mean DCA Coupling J (Position × Position, mean over AAs)")
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "hiv_J_mean_heatmap.png"))
        plt.show()
    else:
        print(f"File not found: {J_path}")