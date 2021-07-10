# Glaucoma Eye Drop Recognition
Note: built with a team [Eric Chou, Ludovico Verniani]

Glaucoma is the leading cause of blindness in the United States with more than 4 million people
affected. There are several types of glaucoma including closed angle, open angle,
pseudoexfoliation, etc. ranging from acute to chronic. In short, there lacks a single true solution
that would be able to solve every type of glaucoma. There are, however, treatments. The most
effective treatment is of course surgery, but there are also medications that reduce pressure on
the eye. An example of such treatment are lasers, which increase the eye’s drainage angle. An important
aspect of such treatments is that most of these are preventative treatments. The problem lies in
the fact that symptoms of glaucoma at its early stages are basically unnoticeable. Once
symptoms occur, it is much too late since the damage caused by it is irreversible. So, the
problem is as follows: if a patient were to get the disease, they will need assistance in
recognizing certain bottles of medication.

Our solution: an iOS app that can capture an image of an unidentified eye drop bottle, identify it
in real-time, and read it out loud to the user. This will help those who have bad vision as a result
of glaucoma guarantee they are in control of their situation and are not taking the wrong
medication.

The dataset is from the UCI Health Gavin Herbert Eye Institute. It contains 4000 pictures of
different types of glaucoma medicine. It is important to note that all the collected data consists
of bottled medicine, not vials, blister packs, sachets, etc. The data are all labeled based on what
medicine the bottle contains. There are five different eye drop medications and therefore five
different categories. They are as follows:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47437080/125173615-0f5b5880-e175-11eb-8519-485eb0bf492c.png" width="600">
</p>

We decided to build on MobileNetV2 as our base model. Becuase we wanted it to be efficient on a mobile app, its speed and accuracy was the right fit for us.
We used TensorFlow as our main machine learning framework. More specifically,
we used the higher level library that runs on top of it, Keras. We wanted a framework that
would allow us to build, train, and evaluate rather quickly. Keras also provided various neural
net architectures that we thought were useful in our situation such as MobileNetV2. Finally, we added layers of Global Average Pooling, Dropout, and Dense. These provided some
ways to combat overfitting. We used average pooling instead of max pooling so that the model
looked at images that were smoother and would return features that were more representative
of the image rather than selecting pixels based on pixel brightness. Note: Our code also does some
preprocessing on the images to expand the variability.

Our results are as follows:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47437080/125173831-2d758880-e176-11eb-8b8c-2229f034092f.png" width="650">
</p>
Note: results include our final model and models we built previously.

