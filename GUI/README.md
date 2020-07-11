GUI
----

We make a simple GUI (inspired heavily from SMS-Tools), which does three things;
- *fixedP* : Reconstruction of fixed pitch notes
- *varP* : Reconstruction of continuously varying pitch contour audio 
- *Generation* : Synthesis of notes in two ways; (1). Provide Vibrato parameters or (2). Provide an external monophonic audio file whose pitch contour will be extracted and a violin rendering corresponding to the extracted pitch will be synthesized.

In the first two, the GUI asks for the PyTorch trained network (the \textit{.pth} file which contains the weights and network description), and the input audio which has to be reconstructed by the network. In the third case, you have to provide the external monophonic audio file whose pitch has to be extracted, or the vibrato parameters for the generated audio.
