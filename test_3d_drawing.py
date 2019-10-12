from prepare_lyft_data import extract_single_box, \
    parse_train_csv,level5data,extract_boxed_clouds
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

from viz_util_for_lyft import draw_lidar_simple

def plot_demo_lyft_lidar():
    train_df = parse_train_csv()

    box, sample_data_token = extract_single_box(train_df, 0)

    ldp_file_path = level5data.get_sample_data_path(sample_data_token)
    lcdp = LidarPointCloud.from_file(ldp_file_path)

    draw_lidar_simple(lcdp.points[0:3,:])

    input()

def plot_cloud_in_box():
    import pickle
    pfile="/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts/val_pc.pickle"

    with open(pfile,'rb') as fp:
        item=pickle.load(fp)
        print(type(item))

    #pc = np.loadtxt('mayavi/kitti_sample_scan.txt')
    fig = draw_lidar_simple(item['pcl'][3])
    #mlab.savefig('pc_view.jpg', figure=fig)
    input()

def plot_demo_random():
    pass

#plot_demo_lyft_lidar()
plot_cloud_in_box()