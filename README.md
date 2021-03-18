# Deep Q Learning TicTacToe

[![Language](https://img.shields.io/badge/language-python-blue.svg?style=flat
)](https://www.python.org)
[![Module](https://img.shields.io/badge/module-pygame-brightgreen.svg?style=flat
)](http://www.pygame.org/news.html)
[![Release](https://img.shields.io/badge/release-v1.0-orange.svg?style=flat
)](https://amaynez.github.io/TicTacToe/)

<div style="text-align:center"><a href="https://amaynez.github.io/TicTacToe/"><img src='/media/Game_Screen.png' width="310" height="300"></a></div>

This program implements a PyGame TicTacToe that can be played by two humans, by a human vs. an algorithmic AI, and a human vs. a Neural Network trained by playing against the algorithmic AI.

The training algorithm uses Deep Mind's DQN recommendations:
- A replay experience memory was implemented capped at 250,000 experiences
- Batches of random experiences from the replay memory are used for every training round
- A secondary neural network was used to calculate the future Q values and then it was updated with the main network's weights every 10 games

For more information: https://amaynez.github.io/TicTacToe/
