import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def draw_bounding_boxes(I, anns, categories, idx):
    """
            Display the specified annotations.
            :param anns (array of object): annotations to display
            :return: None
            """

    # merge id with idNames
    category_id = [cat1['id'] for cat1 in categories]
    category_names = [cat1['name'] for cat1 in categories]
    array1 = np.array([category_id, category_names])

    # get picture id
    info_id = [cat1['category_id'] for cat1 in anns]
    group_id = dict(np.nditer(array1, flags=['external_loop'], order='F'))

    plt.imshow(I)
    plt.axis('off')
    if len(anns) == 0:
        return 0
    if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
        datasetType = 'instances'
    elif 'caption' in anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')
    if datasetType == 'instances':
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in anns:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h],
                    [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            color.append(c)
            if ann["category_id"] in info_id:
                plt.text((ann["bbox"][0]),
                         (ann["bbox"][1]),
                         group_id[str(ann["category_id"])],
                         fontsize=8, color="black")

        # p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        # ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

        plt.savefig('/images/inference_examples_{}.png'.format(idx))

