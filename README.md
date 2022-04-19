# submatter
OpenCV AI competition 2021 project code repository.

The goal of this project is to create a proof of concept showing that useful metrics can be tracked reliably using the OAK-D AI camera during simulated Applied Behavior Analysis therapy of children with Autism Spectrum Disorder (ASD). Research has shown that head movement can be a predictive indictor of ASD severity but an underlying problem remains; tracking metrics regularly, accurately and without bias is difficult for
human beings.

Three variations of the camera firmware are included in this repository, in order of increasing complexity:
1. landmarking_main.py -> Naive approach relying on facial landmarks and geometry of a reference head.
1. main.py             -> High accuracy approach deploying pre-trained neural network [Hopenet](https://github.com/OverEuro/deep-head-pose-lite)
1. no_host_main.py     -> Transfer learning to modify Hopenet to run with no host, improving framerate from 2 FPS to 20 FPS.
