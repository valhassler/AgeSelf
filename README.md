# Age_Self

- This repo is used to detect the faces of humans in videos

- One can train a resnet that classifies human faces in terms of age and gender using lagenda (training_resnet_on_lagenda_age_gender.py)
- Use a face detector pretrained (retinanet (https://github.com/biubug6/Pytorch_Retinaface)) to then combine with the face classifier to do the according classifications (eval_face.py) (in parallel is just that multiple versions run on the same node with distributed cpus)
- Use eyetracking data collected vias Pupilcore or Neon eyetracker to combine with the other data and then show if one looks in the face of some person (filter_faces_eyetracking_Neon_Pupilcore.py)
- Also a weak form of tracking is applied with just IoU to make the classification of videos more consistent (filter_faces_eyetracking_Neon_Pupilcore.py)

- Additionaly there is right now a ipynb that can help to create additional training images for faces (cut_out_faces.ipynb)


- There is also the case where I just did overfitted detection in the Wortschatzinsel for grown up and child these results are right now here. Probably change in future. There is also some stuff how to deal with the cvat API:
/mnt/lustre-emmy-hdd/usr/u11216/datasets/Wortschatzinsel/object_detection_train/5_Annotate_video.ipynb