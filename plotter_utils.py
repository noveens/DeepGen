import math
import numpy as np

def load_matplotlib():
	import matplotlib
	import matplotlib.pyplot as plt
	import seaborn as sns

	sns.set()

	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['lines.linewidth'] = 4
	matplotlib.rcParams['axes.linewidth'] = 2
	matplotlib.rcParams['lines.markersize'] = 14
	matplotlib.rcParams['ps.fonttype'] = 42
	# matplotlib.rcParams['xtick.labelsize'] = 18
	# matplotlib.rcParams['ytick.labelsize'] = 18
	matplotlib.rcParams['axes.xmargin'] = 0.1
	matplotlib.rcParams['axes.ymargin'] = 0.1
	# matplotlib.rc('font', **{'weight': 'normal', 'size': 16})
	# plt.style.use('dark_background')

	ALPHA = 0.6
	COLOR_PALETTE = list(map(lambda x: list(map(lambda y: float(y) / 255.0, x)), [
		(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
		(44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
		(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
		(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
		(188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
	]))

	return plt, ALPHA, COLOR_PALETTE

'''
images: [ N x h x w x 3 ]
return matplotlib figure
'''
def get_combined_image_plot(images, images_per_row):
	plt, ALPHA, COLOR_PALETTE = load_matplotlib()

	images = (images + 1) / 2 ### Un-norm

	N, *_ = images.shape
	total_rows = math.ceil(N / images_per_row)

	FIGSIZE = (2.2 * images_per_row, 1.6 * total_rows)
	fig, ax = plt.subplots(total_rows, images_per_row, sharex = 'row', sharey = 'row', figsize = FIGSIZE)

	for i in range(N):
		if images_per_row == 1: ax_image = ax[i]
		else: ax_image = ax[(i // images_per_row), i % images_per_row]

		ax_image.imshow(images[i, ...].transpose((1, 2, 0)), interpolation='nearest', aspect='auto')
		ax_image.axis('off')
	
	fig.tight_layout(pad = 0.5)
	return fig

'''
images: [ N x h x w x 3 ]
labels: [ N x num_classes ]
return matplotlib figure
'''
def get_combined_image_plot_with_labels(hyper_params, images, labels):
	plt, ALPHA, COLOR_PALETTE = load_matplotlib()
	images_per_row = 10

	N, num_classes = labels.shape
	total_rows = 2 * math.ceil(N / images_per_row)
	height_ratios = [ 1 if i%2 else 3 for i in range(total_rows) ]
	# width_ratios = [ 1 if i%2 else 3 for i in range(images_per_row) ]

	FIGSIZE = (2.2 * images_per_row, 1.6 * total_rows)
	fig, ax = plt.subplots(total_rows, images_per_row, gridspec_kw={'height_ratios': height_ratios}, sharex = 'row', sharey = 'row', figsize = FIGSIZE)

	for i in range(N):
		ax_image = ax[2*(i // images_per_row), i % images_per_row]
		ax_label = ax[(2*(i // images_per_row)) + 1, i % images_per_row]

		ax_image.imshow(images[i, ...], interpolation='nearest', aspect='auto')
		ax_image.axis('off')

		def norm(y): 
			# Un one-hot
			trans = np.array(y) + (1. / num_classes)

			# Softmax
			return np.exp(trans) / sum(np.exp(trans))

		x = list(range(num_classes))
		ax_label.bar(x, norm(labels[i]))
		ax_label.set_xticks(x)
		if i >= N - images_per_row: ax_label.set_xticklabels(hyper_params['class_labels'], rotation = 45)
		else: ax_label.set_xticklabels([ None ] * len(hyper_params['class_labels']))
		# ax_label.axis('off')
	
	fig.tight_layout(pad = 0.5)
	# plt.subplots_adjust(wspace=-0.2)
	return fig
