# BabyBench Wiki

To know more about the *IEEE* ICDL 2025 BabyBench Competition, see [here]().

## Index

- [Home](README.md)

- [About MIMo](mimo.md)
  
- [Training environments](environments.md)
  
- [Submission](submission.md)
  
- [Evaluation](evaluation.md)
  
- [FAQ](faq.md)
  
- [Troubleshooting](troubleshooting.md)

The early stages of human development are characterized by rich sensorimotor exploration where infant engage in self-touch, self-reach and spontanoous movements that contribute to body awareness and motor control (Rochat 1998). These behaviors are observed withi the first months of life and are fundamental for the emergence of the calibration of proprioception and motor coordination. While infant infants development has been extensively studied in psychology and neuroscience, replicating these behaviors on a humanoid robotic platform remains a major challenge.

This research challenge invites participants to design and implement mechanisms that enable a baby-sized humanoid agent, equipped with tactile skin to autonomously generate self-touching and self-reaching behaviors similar to those seen in human infants. The objective here is to model developmental principles such as learning how to explore one's body, building and exploiting sensorimotor loops and intrinsic motivation that drives body-oriented actions in human infants (Oudeyer & Smith, 2016).

Successful solution should demonstrate emergent behaviors where the agent, through self interaction, discovers, affordances of movements and refines motor skills without explicitly programmed trajectories.
Rochat, P. (1998). Self-perception and action in infancy. Experimental Brain Research, 123(1-2), 102-109.
Oudeyer, P. Y., & Smith, L. B. (2016). How evolution may work through curiosity-driven developmental process. Topics in Cognitive Science, 8(2), 492-502.
Participants in this research challenge will work within a custom-designed MIMo (Multi-modal Infant-like Model) environment, API of Gymnasium environments implemented in MuJoCo (Multi-Joint dynamics with Contact) . This simulation environment has been specifically tailored to support the study of early infant-like behaviors, such as self-touch and self-reach, by providing a realistic, physics-based humanoid model equipped with tactile sensing.
