
import os
import scipy
import skimage.io
import numpy as np

def load_reference_image():
    """
    :return: load and return the reference images
    """
    reference_image = {1: ["13.jpg", "1075.jpg", "257.jpg", "1115.jpg", "1562.jpg"],
                       2: ["278.jpg", "910.jpg","102.jpg", "1021.jpg", "1577.jpg"],
                       3: ["1072.jpg", "974.jpg", "617.jpg", "1081.jpg", "1477.jpg"],
                       4: ["293.jpg", "1629.jpg", "767.jpg", "1014.jpg", "1096.jpg"],
                       5: ["761.jpg", "659.jpg","1049.jpg", "1049.jpg", "937.jpg"],
                       6: ["77.jpg", "552.jpg", "112.jpg", "1059.jpg", "1013.jpg"],
                       7: ["197.jpg", "587.jpg", "537.jpg", "108.jpg", "1603.jpg"],
                       8: ["1207.jpg", "1002.jpg", "1006.jpg", "1107.jpg", "1279.jpg"],
                       9: ["568.jpg", "1100.jpg", "1004.jpg", "1036.jpg", "690.jpg"],
                       10: ["884.jpg", "1158.jpg", "1213.jpg", "104.jpg", "1400.jpg"],
                       11: ["101.jpg","730.jpg", "1151.jpg", "1094.jpg", "1389.jpg"]}
    random_choice_image = {}
    for key, values in reference_image.items():
        random_choice_image[key] = reference_image[key][:3] #np.random.choice(reference_image[key], 3)
    reference_image_list = []
    for r_label in range(1, 12):
        for r_image in random_choice_image[r_label]:
            image = skimage.io.imread(os.path.join(os.getcwd(), "images", str(r_label), r_image))
            image = scipy.misc.imresize(image, (224, 224))
            reference_image_list.append(image)
    reference_image_array = np.array(reference_image_list)

    return reference_image_array