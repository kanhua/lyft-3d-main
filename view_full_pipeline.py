from viz_util_for_lyft import PredViewer
from prepare_lyft_data_v2 import load_train_data
from test_data_loader import load_test_data
import matplotlib.pyplot as plt

# TODO: Need to add prepare_lyft_data_v2_rgb_test.FrustumRGBTestCase.test_plot_frustums to complete full pipeline

def plot_prediction_data():
    lyftd = load_test_data()

    pv = PredViewer(pred_file="test_pred.csv", lyftd=lyftd)

    # test_token = lyftd.sample[2]['token']
    test_token = pv.pred_pd.index[100]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    pv.render_camera_image(ax[0], sample_token=test_token, prob_threshold=0.1)

    pv.render_lidar_points(ax[1], sample_token=test_token, prob_threshold=0.1)

    fig.savefig("./artifact/camera_top_view.png", dpi=600)

    pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.1)


if __name__ == "__main__":
    plot_prediction_data()

    import mayavi.mlab as mlab

    mlab.savefig("./artifact/predicted_3d_pc.png")

    input()
