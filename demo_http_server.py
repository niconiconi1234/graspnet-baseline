from flask import Flask, request
from waitress import serve
import numpy as np
from demo import get_net, get_grasps
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import open3d as o3d
import torch
import argparse
from graspnetAPI import Grasp

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01,
                    help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

app = Flask(__name__)
grasp_net = get_net()


def get_and_process_data2(color,
                          depth,
                          workspace_mask,
                          intrinsic,
                          factor_depth,
                          camera_width=1280.0,
                          camera_height=720.0):
    # generate cloud
    camera = CameraInfo(
        camera_width, camera_height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(
            len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(
            len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(
        cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def grasp2dict(obj: Grasp):
    if isinstance(obj, Grasp):
        return {
            'depth': obj.depth,
            'grasp_array': obj.grasp_array.tolist(),
            'height': obj.height,
            'object_id': obj.object_id,
            'rotation_matrix': obj.rotation_matrix.tolist(),
            'score': obj.score,
            'translation': obj.translation.tolist(),
            'width': obj.width
        }
    return {}


@app.route('/grasp', methods=['POST'])
def grasp():

    def b64_to_np_image(b64_string):
        from PIL import Image
        import base64
        from io import BytesIO
        b64_string = ',' in b64_string and b64_string.split(',')[1] or b64_string
        im = Image.open(BytesIO(base64.b64decode(b64_string)))
        return np.array(im)

    color = request.json['color']
    depth = request.json['depth']
    workspace_mask = request.json['workspace_mask']
    intrinsics = request.json['meta']['intrinsic_matrix']
    factor_depth = request.json['meta']['factor_depth']
    camera_width = request.json['meta']['camera_width']
    camera_height = request.json['meta']['camera_height']

    color = b64_to_np_image(color)
    depth = b64_to_np_image(depth)
    workspace_mask = b64_to_np_image(workspace_mask)

    end_points, cloud = get_and_process_data2(color, depth, workspace_mask, intrinsics, factor_depth,
                                              camera_height=camera_height, camera_width=camera_width)

    gg = get_grasps(grasp_net, end_points)

    grasps = []
    n = len(gg.translations)
    for i in range(n):
        grasps.append(grasp2dict(gg[i]))

    grasps.sort(key=lambda x: x['score'], reverse=True)
    return grasps


def main():
    port = 50053
    serve(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
