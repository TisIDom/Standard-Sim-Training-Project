import os
import tensorflow as tf

def compute_errors(gt, pred, print_res=False):
  """Evaluates depth metrics using TensorFlow operations"""

  abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0

  # Remove invalid predictions (> 100) and set them to max valid value
  pred = tf.where(pred > 100, tf.reduce_max(pred), pred)
  pred = tf.where(pred == -1, tf.reduce_max(pred), pred)

  # Same for ground truth
  gt = tf.where(gt > 100, tf.reduce_max(gt), gt)
  gt = tf.where(gt == -1, tf.reduce_max(gt), gt)

  # Calculate non-zero mask
  nonzero_mask = tf.cast(gt > 0, tf.float32)

  # Filter valid predictions and ground truth based on mask
  valid_gt = tf.math.multiply(gt, nonzero_mask)
  valid_pred = tf.math.multiply(pred, nonzero_mask)

  # Check for empty valid data and skip
  valid_count = tf.math.count_non_zero(valid_gt)
  if valid_count == 0:
    return [0.0 for _ in range(6)]

  # Calculate threshold metric
  thresh = tf.maximum(valid_gt / valid_pred, valid_pred / valid_gt)
  a1 += tf.reduce_mean(tf.cast(thresh < 1.25, tf.float32))
  a2 += tf.reduce_mean(tf.cast(thresh < 1.25**2, tf.float32))
  a3 += tf.reduce_mean(tf.cast(thresh < 1.25**3, tf.float32))

  # Absolute difference, relative difference, squared relative difference
  abs_diff = tf.reduce_mean(tf.abs(valid_gt - valid_pred))
  abs_rel = tf.reduce_mean(tf.abs(valid_gt - valid_pred) / valid_gt)
  sq_rel = tf.reduce_mean(((valid_gt - valid_pred)**2) / valid_gt)

  # Print individual results if specified
  if print_res:
    res_list = tf.reduce_mean(tf.abs(valid_gt - valid_pred), axis=0).numpy()
    print(res_list)

  # Return metrics as list
  return [abs_rel, abs_diff, sq_rel, a1, a2, a3]

if __name__ == "__main__":
  pred_root = '/app/MiDaS/output'
  gt_root = '/app/renders_multicam_diff_1'

  pfm_images = [f for f in os.listdir(pred_root) if f.endswith(".pfm")]
  gt_images = [f[:-4]+'-depth0001.exr' for f in pfm_images]

  preds = []
  gts = []
  count_zeros = []
  count_zeros_percentages = []
  for i in range(len(pfm_images)):
    # Load prediction using tf.io.read_file
    pred_path = os.path.join(pred_root, pfm_images[i])
    pred_img = tf.io.read_file(pred_path)
    # Assuming PFM files contain single-channel float32 data
    pred_img = tf.decode_raw(pred_img, tf.float32)
    pred_img = tf.reshape(pred_img, (-1,))  # Reshape to 1D for processing

    # Preprocess prediction (same as before)
    pred = compute_errors.preprocess_prediction(pred_img)

    # Load ground truth using imageio
    gt_path = os.path.join(gt_root, gt_images[i])
    gt_img = tf.io.read_file(gt_path)
    gt_img = tf.image.decode_image(gt_img, dtype=tf.float32)[:,:,0]  # Assuming single channel

    # Preprocess ground truth (same as before)
    gt = compute_errors.preprocess_prediction(gt_img)

    # Check for zeros and append data
    if tf.math.count_non_zero(gt) > 0:
      count_zeros_percentages.append(tf.reduce_mean(tf.cast(gt == 0, tf.float32)))
      count_zeros.append(gt_images[i])

    preds.append(pred)
    gts.append(gt)

  # Print results (converted to TensorFlow operations)
  print("Num gt images with zeros ", len(count_zeros))
  print("Fraction of pixels equal zero ", tf.reduce_mean(count_zeros_percentages))
  print(compute_errors(tf.stack(preds), tf.stack(gts)))