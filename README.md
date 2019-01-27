# player
DQN on TH - IN

This is just a snapshot

My setup is to run the game on windows, sending frames/metadata to a local Ubuntu server with a better GPU.
The GPU runs the DQN over the frame and sends the recommendation back to windows.

Currently game_interfacer sends commands to the active screen. Th uses DirectX and ignores send_input.

Start dqn.py, open the game and select the level, then run game_interfacer.

requires cnn_vis for kernel activation/deconvolution
