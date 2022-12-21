# hdf5 supports large, complex, heterogeneous data
# dataset API supports hdf5 files for better results (??)

# load pcd's and extract the points
def generate_hdf5_dataset_with_padding(path, run_name, hdf5_filename):
	# Build main path
	path = join(path, run_name)
	
	# Get files
	jpgs = sorted(glob.glob(path+'/jpg/*.jpg'))
	pcds = sorted(glob.glob(path+'/pcd/*.pcd'))

	# Open HDF5 file in write mode
	with h5py.File(hdf5_filename, 'w') as f:
		
		images = []
		point_clouds = []
		
		# Determine the size of largest point cloud for padding
		max_size = 0

		for i, jpg in enumerate(jpgs):

			base_name = jpg[jpg.rfind('/')+1:jpg.find('.jpg')]

			# Load the image
			image = cv2.cvtColor(cv2.imread(jpgs[i]), cv2.COLOR_BGR2RGB) 
			images.append(image)

		
			# Load the point cloud
			cloud = o3d.io.read_point_cloud(pcds[i])
			points= np.asarray(cloud.points)
			point_clouds.append(points)
			
			# Keep track of largest size
			if points.shape[0] > max_size:
				max_size = points.shape[0]
			
			if ((i+1) % 1000 == 0):
				print('Processed ',(i+1),' pairs of files.')
				
		print('max size ', max_size)
		print('padding ...')

		# Pad the point clouds with 0s
		padded_point_clouds = []
		for points in point_clouds:
			pad_amount = max_size - points.shape[0]
			
			points_padded = np.pad(points, ((0, pad_amount),(0, 0)), 'constant', constant_values=(0, 0))
			padded_point_clouds.append(points_padded)

		# Create an images and a point clouds dataset in the file
		f.create_dataset('images', data = np.asarray(images))
		f.create_dataset('point_clouds', data = np.asarray(padded_point_clouds))

# Dataset API

def resize_and_format_data(points, image):
	# Sample a random number of points
	idxs = tf.range(tf.shape(points)[0])
	ridxs = tf.random.shuffle(idxs)[:SAMPLE_SIZE]
	points = tf.gather(points, ridxs)

	# Normalize pixels in the input image
	image = tf.cast(image, dtype=tf.float32)
	image = image/127.5
	image -= 1

	return points, image

def get_training_dataset(hdf5_path):
	# Get the point clouds
	x_train = tfio.IODataset.from_hdf5(hdf5_path, dataset='/point_clouds')
	# Get the images
	y_train = tfio.IODataset.from_hdf5(hdf5_path, dataset='/images')
	# Zip them to create pairs
	training_dataset = tf.data.Dataset.zip((x_train,y_train))
	# Apply the data transformations
	training_dataset = training_dataset.map(resize_and_format_data)

	# Shuffle, prepare batches, etc ...
	training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
	training_dataset = training_dataset.batch(BATCH_SIZE)
	training_dataset = training_dataset.repeat()
	training_dataset = training_dataset.prefetch(-1)

	# Return dataset
	return training_dataset
