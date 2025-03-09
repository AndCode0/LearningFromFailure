import numpy as np
from Data.CelebA import CustomCelebA
from torchvision import transforms
from torch.utils.data import DataLoader

print_matrices = False


# Set up data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize((0.5061, 0.4254, 0.3828), (0.2875, 0.2744, 0.2729)),
])

# Load CelebA dataset
dataset = CustomCelebA(root='C:\\Users\\aconte\\Desktop', split='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

imgs, _ = next(iter(dataloader))
print("Batch mean", imgs.mean(dim=[0,2,3]))
print("Batch std", imgs.std(dim=[0,2,3]))

NUM_IMAGES = 4
images = [dataset[idx][0] for idx in range(NUM_IMAGES)]
orig_images = [Image.fromarray(train_dataset.data[idx]) for idx in range(NUM_IMAGES)]
orig_images = [test_transform(img) for img in orig_images]

img_grid = torchvision.utils.make_grid(torch.stack(images + orig_images, dim=0), nrow=4, normalize=True, pad_value=0.5)
img_grid = img_grid.permute(1, 2, 0)

plt.figure(figsize=(8,8))
plt.title("Augmentation examples on CIFAR10")
plt.imshow(img_grid)
plt.axis('off')
plt.show()
plt.close()

assert print_matrices, 'If you don\'t want matrices you\'re done! GJ'
# Initialize variables
all_images = []

count = 0
sum_pixels = 0
sum_squared_pixels = 0


for i, (images, _) in enumerate(dataloader):
    # Convert to numpy for calculation
    batch = images.numpy()
    batch_size = batch.shape[0]
    
    # Flatten the images to simplify calculations
    flat_batch = batch.reshape(batch_size, -1)
    
    # Update count
    count += batch_size
    
    # Update sum of pixels
    sum_pixels += np.sum(flat_batch, axis=0)
    
    # Update sum of squared pixels
    sum_squared_pixels += np.sum(flat_batch ** 2, axis=0)
    
    print(f"Processed batch {i+1}, total images: {count}")

# Calculate mean
mean_pixels = sum_pixels / count

# Calculate variance and std
# Var(X) = E[X^2] - E[X]^2
variance_pixels = (sum_squared_pixels / count) - (mean_pixels ** 2)
std_pixels = np.sqrt(variance_pixels)

# Reshape back to image format
channels, height, width = images[0].shape
mean_face = mean_pixels.reshape(channels, height, width)
std_face = std_pixels.reshape(channels, height, width)


np.savetxt('mean_face0.txt', mean_face[0], delimiter=',')
np.savetxt('mean_face1.txt', mean_face[1], delimiter=',')
np.savetxt('mean_face2.txt', mean_face[2], delimiter=',')

np.savetxt('std_face0.txt', std_face[0], delimiter=',')
np.savetxt('std_face1.txt', std_face[1], delimiter=',')
np.savetxt('std_face2.txt', std_face[2], delimiter=',')