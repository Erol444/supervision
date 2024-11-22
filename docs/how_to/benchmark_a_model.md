# Benchmark a Model

## Overview

Have you ever trained multiple object detection models and wondered which one performs best on your specific use case? Or maybe you've downloaded a pre-trained model and want to verify its performance on your dataset? Model benchmarking is essential for making informed decisions about which model to deploy in production.

This guide will show an easy way to benchmark your results using `supervision`. It will go over:

1. [Loading a dataset](#loading-a-dataset)
2. [Loading a model](#loading-a-model)
3. [Benchmarking Basics](#benchmarking-basics)
4. [Visual Benchmarking](#visual-benchmarking)
5. [Metric: Mean Average Precision (mAP)](#metric-mean-average-precision-map)
6. [Metric: F1 Score](#metric-f1-score)
7. [Bonus: Model Leaderboard](#bonus-model-leaderboard)

This guide applies to object detection, instance segmentation, and oriented bounding box models (OBB).

## Loading a Dataset

Suppose you start with a dataset. Perhaps you found it on [Universe](https://universe.roboflow.com/); perhaps you [labeled your own](https://roboflow.com/how-to-label/yolo11). In either case, this guide assumes you have a dataset with labels in your [Roboflow Workspace](https://app.roboflow.com/).

Let's use the following libraries:

- `roboflow` to manage the dataset and deploy models
- `inference` to run the models
- `supervision` to evaluate the model results

```bash
pip install roboflow inference supervision
```

Here's how you can download a dataset:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="<YOUR_API_KEY>")
project = rf.workspace("<WORKSPACE_NAME>").project("<PROJECT_NAME>")
dataset = project.version(<DATASET_VERSION_NUMBER>).download("<FORMAT>")
```

In this guide, we shall use the [car part segmentation dataset](https://universe.roboflow.com/alpaco5-f3woi/part-autolabeld).

```python
from roboflow import Roboflow

rf = Roboflow(api_key="yqBDUaljWHfDkS7xGDjg")
project = rf.workspace("alpaco5-f3woi").project("part-autolabeld")
dataset = project.version(5).download("yolov11")
```

This will create a folder called `part-autolabeld` with the dataset in the current working directory, with `train`, `test`, and `valid` folders and `data.yaml` file.

## Loading a Model

Let's load a single model, and see how to evaluate it. To evaluate another, simply return to this step.

=== "Pretrained Model"

    Roboflow supports a range of state-of-the-art [pre-trained models](https://inference.roboflow.com/quickstart/aliases/) for object detection, instance segmentation, and pose tracking. You don't even need an API key!

    Let's load such a model with inference [`inference`](https://inference.roboflow.com/).

    ```python
    from inference import get_model

    model = get_model(model_id="yolov11s-640")
    ```

=== "Trained on Roboflow Platform"

    You can train and deploy a model without leaving the Roboflow platform. See this [guide](https://docs.roboflow.com/train/train/train-from-scratch) for more details.

    To load a model, you can use inference:

    ```python
    from inference import get_model

    model_id = "<PROJECT_NAME>/<MODEL_VERSION>"
    model = get_model(model_id=model_id)
    ```

=== "Locally Trained Model"

    The following code applies to segmentation, but can also be used for object detection and oriented bounding box models, if you change `task` and `model` arguments.

    ```bash
    pip install ultralytics
    yolo task=segment mode=train model=yolo11s-seg.pt data=part-autolabeld/data.yaml epochs=10 imgsz=640
    ```

    Once the model is trained, you can deploy it to Roboflow, making it available anywhere.

    Note: if using other model types, change to `-obb` or remove suffix in `model_type` and replace `segment` with `obb`or `detect`. Multiple runs also produce multiple folders such as `segment`, `segment1`, `segment2`, etc.

    ```python
    project.version(dataset.version).deploy(
        model_type="yolov11-seg", model_path=f"runs/segment/train/weights/best.pt"
    )

    from inference import get_model
    model_id = project.id.split("/")[1] + "/" + dataset.version
    model = get_model(model_id=model_id)
    ```

## Benchmarking Basics

Evaluating your model requires careful selection of the dataset. Which images should you use?Let's go over the different subsets of a dataset.

- **Training Set**: This is the set of images used to train the model. Since the model learns to maximize its accuracy on this set, it should **never** be used for validation - the results will seem unrealistically good.
- **Validation Set**: This is the set of images used to validate the model during training. Every Nth training epoch, the model is evaluated on the validation set. Often the training is stopped once the validation loss stops improving. Therefore, even while the images aren't used to train the model, it still influences the training outcome.
- **Test Set**: This is the set of images kept aside for model testing. It is exactly the set you should use for benchmarking. If the dataset was split correctly, none of these images would be shown to the model during training.

Therefore, the `test` set is the best choice for benchmarking.
Several other problems may arise:

- **Data Contamination**: It's possible that the dataset was not split correctly and some images from the test set were used during training. In this case, the results will be overly optimistic. It also covers the case where **very similar** images were used for training and testing - e.g. those taken in the same environment.
- **Missing Test Set**: Some datasets do not come with a test set. In this case, you should collect and [label](https://roboflow.com/annotate) your own data. Alternatively, a validation set could be used, but the results could be overly optimistic. Make sure to test in the real world asap.

<!-- TODO: continue from here -->

## Visualizing Predictions

The first step in evaluating your model’s performance is to visualize its predictions. This gives an intuitive sense of how well your model is detecting objects and where it might be failing.

- **Running Visualizations**: Below is a code example that demonstrates visualizing model predictions:

  ```python
  from supervision import visualize_predictions

  visualize_predictions(model, dataset_path='path/to/images')
  ```

  > **Highlighted Line**: Use `visualize_predictions` to easily generate visual outputs showing the model's predictions against your test images.

- **Tips for Visual Inspection**: Randomly sample images from the dataset and visualize predictions. Ensure all classes are correctly identified and the bounding boxes align with expected outcomes.

  - For different types of tasks (e.g., **object detection**, **masking**, **oriented bounding boxes (OBB)**), refer to our [annotator documentation](#link-to-annotator-docs) to choose the right tool for your visualizations.

## Section 4: Evaluating Performance Metrics

### Mean Average Precision (mAP)

**mAP** is a crucial metric for evaluating object detection models. It provides a score representing how well the model identifies objects across various classes and IoU (Intersection over Union) thresholds.

- **Calculate mAP**:

  ```python
  from supervision import calculate_map

  map_score = calculate_map(predictions, ground_truth)
  print(f'mAP: {map_score}')
  ```

  > **Highlighted Line**: The `calculate_map` function computes the mean average precision across your dataset, providing an overall score of model quality.

> **Why mAP?**: It is the most popular metric for object detection tasks as it captures both localization and classification accuracy.

### F1 Score

The **F1 score** is another useful metric, especially when dealing with an imbalance between false positives and false negatives. It’s the harmonic mean of **precision** (how many predictions are correct) and **recall** (how many actual instances were detected).

- **Calculate F1 Score**:

  ```python
  from supervision import calculate_f1

  f1_score = calculate_f1(predictions, ground_truth)
  print(f'F1 Score: {f1_score}')
  ```

  > **Highlighted Line**: The `calculate_f1` function helps provide a balance between precision and recall, especially useful when one value disproportionately affects the evaluation.

> **Precision & Recall**: Briefly, **precision** helps minimize false positives, while **recall** ensures fewer false negatives. F1 combines both into a single metric.

### Is mAR Worth Adding?

**Mean Average Recall (mAR)** is not always needed, but if you’re particularly interested in understanding how well your model can find all relevant instances, it may be valuable.

- **mAR Example**:

  ```python
  from supervision import calculate_mar

  mar_score = calculate_mar(predictions, ground_truth)
  print(f'mAR: {mar_score}')
  ```

  > **Highlighted Line**: While optional, calculating `mAR` can offer insights into the model’s recall capabilities, especially if minimizing false negatives is a priority.

## Bonus Section: Model Leaderboard

We’ve done a lot of work to benchmark various models on the COCO dataset. Be sure to check out our [COCO leaderboard](#link-to-leaderboard) to see how different models perform and to get a sense of the state-of-the-art results. It's a great place to understand what the leading models can achieve and to compare your own results.

## Conclusion

In this guide, you’ve learned how to set up your environment, train or use pre-trained models, visualize predictions, and evaluate model performance with metrics like mAP, F1 score, and optionally mAR. Benchmarking your models against our COCO leaderboard helps you see how they stack up against others in the community, gain valuable insights, and continuously improve your models. Sharing your results on Universe can foster collaboration and help push the field forward.

For more details, be sure to check out our documentation and join our community discussions.
