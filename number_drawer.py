import pygame
import sys
import numpy as np
import NueralNetwork as nn

#The NN is already trained so no need for any training data or learning rate
#Also I know Neural is spelled wrong but I kinda like it that way
my_nn = nn.NeuralNetwork([28*28, 500, 250, 100, 10], None, None, None)
#Change this and the initilization of the nn to use other pretrained neural networks
my_nn.import_network('network_500_250_100_97.txt')

pygame.init()

size = (500, 525)
white = (255, 255, 255)

screen = pygame.display.set_mode(size)
font = pygame.font.SysFont("arial", 20)

#Background
background = pygame.Surface(screen.get_size())
background.fill(white)
background.convert()

#Erase Button
erase = pygame.Surface((250, 25))
erase.fill((255, 100, 100))
erase.convert()
erase_text = font.render("Erase", True, (0, 0, 0))
erase.blit(erase_text, (100, 0))

def erase_paper(p):
	p.fill(white)

#Number Display
number_disp = pygame.Surface((250, 25))
number_disp.fill((100, 255, 255))
number_disp.convert()

number_text = font.render("You wrote: ", True, (0, 0, 0))
number_disp.blit(number_text, (50, 0))

#Display for number confidence
array_disp = pygame.Surface((500, 50))
array_disp.convert()
array_disp.fill(white)

def update_disp():
	image = pygame.surfarray.array2d(paper)
	
	#Downscale
	downscaled = np.zeros((28, 28))
	for i in range(0, 476, 17):
		for j in range(0, 476, 17):
			downscaled[int(j/17)][int(i/17)] = image[i][j]
	down.fill(white)
	pygame.surfarray.blit_array(down, downscaled)
	
	#flip colors
	data = np.ones(28*28) - (pygame.surfarray.pixels_red(down)/255).reshape(28*28)
	
	#Identify Number
	my_nn.feed_forward(data)
	num = np.argmax(my_nn.layers[-1])
	
	#Update Number
	number_text = font.render(f'You wrote: {num}', True, (0, 0, 0))
	number_disp.fill((100, 255, 255))
	number_disp.blit(number_text, (50, 0))
	
	#Update Array Display
	array_disp.fill(white)
	for i in range(10):
		value = my_nn.layers[-1][i]
		if (value) > 0:
			pygame.draw.rect(array_disp, (0, 255 * value, 0), (0 + i*50, 0, 50, 50*value))

		else:
			pygame.draw.rect(array_disp, (255 * abs(value), 0, 0), (0 + i*50, 0, 50, 50*value))
		text = font.render(f'{i}', True, (0, 0, 0))
		array_disp.blit(text, (i*50 + 15, 10))

#Paper, the surface we draw on
paper = pygame.Surface((476, 476))
paper.fill((255, 255, 255))
paper.convert()

def draw_paper(coords, p):
	pygame.draw.circle(p, (0, 0, 0), coords, 40)

#The downscaled surface of the paper we feed to the nn
down = pygame.Surface((28, 28))
down.convert()

clock = pygame.time.Clock()

running = True
pressed = False

while running:
	clock.tick(30)
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
			
		elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
			if erase.get_rect(topleft=(0, 500)).collidepoint(pygame.mouse.get_pos()):
				erase_paper(paper)
			else:
				pressed = True
		
		elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			pressed = False
	
	if pressed and paper.get_rect().collidepoint(pygame.mouse.get_pos()):
		draw_paper(pygame.mouse.get_pos(), paper)
	
	update_disp()
	screen.blit(background, (0, 0))
	screen.blit(erase, (0, 500))
	screen.blit(number_disp, (250, 500))
	screen.blit(paper, (0, 0))
	screen.blit(array_disp, (0, 450))
	screen.blit(down, (0, 0))

	
	pygame.display.flip()