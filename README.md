# Age_Self

- This repo is used to detect the faces of humans in videos

- One can train a resnet that classifies human faces in terms of age and gender using lagenda (training_resnet_on_lagenda_age_gender.py)
- Use a face detector pretrained (retinanet (https://github.com/biubug6/Pytorch_Retinaface)) to then combine with the face classifier to do the according classifications (eval_face.py) (in parallel is just that multiple versions run on the same node with distributed cpus)
- Use eyetracking data collected vias Pupilcore or Neon eyetracker to combine with the other data and then show if one looks in the face of some person (filter_faces_eyetracking_Neon_Pupilcore.py)
- Also a weak form of tracking is applied with just IoU to make the classification of videos more consistent (filter_faces_eyetracking_Neon_Pupilcore.py)

- Additionaly there is right now a ipynb that can help to create additional training images for faces (cut_out_faces.ipynb)

- There is also the case where I just did overfitted detection in the Wortschatzinsel (overview camera) for grown up and child these results are created by overfitting a yolov8 pretrained on crowdhuman using PigDetect repo(not public) but it is just a finetuned yolov8 that is used in the end for recognition and classificaton

- There is also some stuff how to deal with the cvat API (just for my info right now):
/datasets/Wortschatzinsel/object_detection_train/5_Annotate_video.ipynb


## How to get it running in conda:

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
conda install \
  pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=11.8 \
  tqdm pillow dlib pandas "numpy<2.0" \
  -c pytorch -c nvidia -c conda-forge -y
pip install decord

clone that repo:
https://github.com/biubug6/Pytorch_Retinaface
go to its folder and do pip install -e . (has a setup.py file)
also do "pip install -e . -v" in the base folder of this repo
```

That is the absic guide