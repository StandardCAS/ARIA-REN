# ARIA-WRE
Artificial Rendering and Image Analysis-with-Wave Reversed Engineering

AI system designed to transform images into sketches and provide a step-by-step replay of the sketching process. ARIA leverages advanced machine learning algorithms to analyze image content, generate corresponding sketches, and create a sequence of steps that mimic the process of sketching by hand. This technology has potential applications in art education, digital art creation, and more. However, it also raises important ethical considerations, particularly regarding the potential for misuse in falsely claiming authorship of AI-generated art. To address this, we propose that any use of ARIA for AI drawings step recovery should be properly tagged as such.

This work, including the ARIA system, is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. 

------------------------

Use the Train2 module for training, change the directories to point to a folder with videos and another with the models saved from running TABLET ONE.py

the folder should look like this

Videos
>a.mp4
>b.mp4

Tablets
>a.npy
>b.npy

Choose any video that is a timelapse playback in mp4 format and run it with TABLET ONE to autosave the tablet model. 

-----------------------

How big is the model? 
4Gb around 10 seconds of timelapse(around 300 frames).

How much space for CPU?
At least 128Gb.

How much space for GPU?
At least 64Gb.
