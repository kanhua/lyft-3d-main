from viz_util_for_lyft import PredViewer, draw_preprocess_results
from prepare_lyft_data_v2 import load_train_data
from test_data_loader import load_test_data
import matplotlib.pyplot as plt


def test_one_sample_token():
    pv = PredViewer(pred_file="train_val_pred.csv", lyftd=load_train_data())
    test_token = pv.pred_pd.index[0]

    fig, ax = plt.subplots(nrows=2, ncols=1)
    pv.render_camera_image(ax[0], sample_token=test_token, prob_threshold=0.4)

    pv.render_lidar_points(ax[1], sample_token=test_token, prob_threshold=0.4)

    plt.show()


def test_3d_lidar_points():
    pv = PredViewer(pred_file="train_val_pred.csv", lyftd=load_train_data())
    test_token = pv.pred_pd.index[0]

    pv.render_3d_lidar_points(sample_token=test_token)


def test_3d_lidar_points_in_camera_coords():
    pv = PredViewer(pred_file="train_val_pred.csv", lyftd=load_train_data())
    test_token = pv.pred_pd.index[0]

    pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.4)


def plot_prediction_data():
    lyftd = load_test_data()
    pv = PredViewer(pred_file="prediction.csv", lyftd=lyftd)

    # test_token = lyftd.sample[2]['token']
    test_token = pv.pred_pd.index[1]

    pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.4)
    from object_classifier import TLClassifier

    tl = TLClassifier()

    draw_preprocess_results(test_sample_token=test_token,
                            lyftd=lyftd,
                            object_classifier=tl)


def plot_train_val_data():
    lyftd = load_train_data()
    pv = PredViewer(pred_file="train_val_pred.csv", lyftd=lyftd, is_train=True)

    # test_token = lyftd.sample[2]['token']
    test_token = pv.pred_pd.index[1]

    pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.4)


if __name__ == "__main__":
    # test_one_sample_token()
    # test_3d_lidar_points()
    # test_3d_lidar_points_in_camera_coords()
    plot_train_val_data()
    import mayavi.mlab as mlab

    mlab.savefig("./artifact/test_mlab_3d.png")

    input()
