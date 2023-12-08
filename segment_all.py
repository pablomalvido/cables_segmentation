import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np
import cv2 as cv
import time
import glob

#User selection
images_folder = "real_images"
#images_folder = "synthetic_images\\close_room_light"

#Initialization of variables
model_list = {'unet':512, 'deeplabv3p':512, 'fcn':512, 'fpn':512,'linknet':512, 'pspnet':480} #Model names and img sizes
metrics = {}
metrics_avg = {}
path_dir = os.path.dirname(os.path.realpath(__file__)) + "\\"+images_folder+"\\"
path_image_dir = path_dir + "images\\"
path_masks_dir = path_dir + "masks\\"
image_list = [f for f in os.listdir(path_image_dir) if os.path.isfile(os.path.join(path_image_dir, f))]


def compare_masks(mask1, mask2):
    # Convert masks to numpy arrays
    mask1 = np.array(mask1, dtype=np.uint8)
    mask2 = np.array(mask2, dtype=np.uint8)

    # Convert masks to binary (0 and 1)
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    true_positives = np.count_nonzero(np.logical_and(mask1, mask2))
    false_positives = np.count_nonzero(np.logical_and(mask1, np.logical_not(mask2)))
    false_negatives = np.count_nonzero(np.logical_and(np.logical_not(mask1), mask2))
    true_negatives = np.count_nonzero(np.logical_and(np.logical_not(mask1), np.logical_not(mask2)))

    return true_positives, false_positives, false_negatives, true_negatives


for model_name in model_list:
  #Load the model
  IMAGE_SIZE = model_list[model_name]
  path_model = os.path.dirname(os.path.realpath(__file__)) + "\\models\\"+model_name
  print("Loading: " + str(model_name) + " model...")
  model = tf.keras.models.load_model(path_model, compile=False)
  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.FalseNegatives()])
  
  #Initialize variables
  metrics[model_name] = {'TP':[], 'FP':[], 'TN':[], 'FN':[], 'IoU':[], 'Dice':[]}
  metrics_avg[model_name] = {}
  directory_predictions = path_dir + "predictions\\" + model_name
  if not os.path.exists(directory_predictions):
      os.makedirs(directory_predictions)

  i=0
  for image_name in image_list:
    #Load image
    ori_x = cv.imread(path_image_dir+image_name, cv.IMREAD_COLOR)
    ori_x = cv.resize(ori_x, (IMAGE_SIZE, IMAGE_SIZE))
    x = ori_x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    #Predict and save result
    start_time = time.time()
    pred_mask = model.predict(x)[0] > 0.5
    prediction_image = tf.keras.preprocessing.image.array_to_img(pred_mask)
    prediction_image2 = tf.keras.preprocessing.image.img_to_array(prediction_image)
    pred_image_cv2 = cv.cvtColor(prediction_image2.astype('uint8'), cv.COLOR_RGB2BGR)
    saved_path = path_dir + "predictions\\" + model_name + "\\" + image_name
    cv.imwrite(saved_path, pred_image_cv2)
    print(str(model_name) + "-" + str(i) + " - Total time: " + str(time.time() - start_time) + "s")
    i+=1

    #Compare prediction with ground truth
    image_gray_segmented = cv.cvtColor(pred_image_cv2, cv.COLOR_BGR2GRAY)
    mask_ground_truth = cv.imread(path_masks_dir+image_name, cv.IMREAD_GRAYSCALE)
    if mask_ground_truth.shape[0] != IMAGE_SIZE:
      mask_ground_truth = cv.resize(mask_ground_truth, (IMAGE_SIZE, IMAGE_SIZE))
    _, binary_image_GT = cv.threshold(mask_ground_truth, 60, 255, cv.THRESH_BINARY)

    #Get metrics
    tp_i, fp_i, fn_i, tn_i = compare_masks(image_gray_segmented, binary_image_GT)
    metrics[model_name]['TP'].append(tp_i)
    metrics[model_name]['FP'].append(fp_i)
    metrics[model_name]['FN'].append(fn_i)
    metrics[model_name]['TN'].append(tn_i)  
    metrics[model_name]['IoU'].append(tp_i/(tp_i+fp_i+fn_i))
    metrics[model_name]['Dice'].append((2*tp_i)/((2*tp_i)+fp_i+fn_i))

  #Results model
  for metric_i in metrics[model_name]:
    metrics_avg[model_name][metric_i] = sum(metrics[model_name][metric_i])/len(metrics[model_name][metric_i])

#Print results summary
for model_name in model_list:
    print(model_name + ":")
    for metric_i in metrics[model_name]:
        print('\t- ' + metric_i + ': ' + str(metrics_avg[model_name][metric_i])+'\n')
    print("---------------------------")