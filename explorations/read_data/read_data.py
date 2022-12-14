from io import BufferedReader
import matplotlib.pyplot as plt
import numpy as np

reverseBytes2Int = lambda arr : (arr[0] << 24) + (arr[1] << 16) + (arr[2] << 8) + arr[3]



def read_img(file: BufferedReader, h, w):
    barr = file.read(h*w)
    res = np.array([bi for bi in barr])
    
    return res.reshape((h, w))

def main():
    
    with open("../../data/train-images-idx3-ubyte", "rb") as file:
        magic = file.read(4)
        print(magic, type(magic), len(magic), magic[1], magic[2], magic[3], type(magic[1]))
        magic = reverseBytes2Int(magic)
        
        num_imgs = file.read(4)
        num_imgs = reverseBytes2Int(num_imgs)
        
        h = file.read(4)
        h = reverseBytes2Int(h)
        
        w = file.read(4)
        w = reverseBytes2Int(w)
        
        print(magic)
        print(f"Data is of size {num_imgs} x {h} x {w}")
        
        digits = [5, 0, 4]
        # for i in range(3):
        #     img = read_img(file, h, w)
        #     print(f"Data: {img[0, 0]} {img[10, 10]}  {img[11, 15]}")
        
        #     plt.imshow(img)
        #     plt.show()
            
            # digit = input("digit: ")
            # digits.append(digit)
    
    print("digits ", digits)
    
    with open("../../data/train-labels-idx1-ubyte", "rb") as file:
        label_magic = file.read(4)
        label_magic = reverseBytes2Int(label_magic)
        
        num_labels = file.read(4)
        num_labels = reverseBytes2Int(num_labels)
        print("Label data: ")
        print(label_magic, ' ', num_labels)
        
        for l in range(3):
            
            label = file.read(1)
            print('raw', label, ' val ', label[0])
            
    
    


if __name__ == '__main__':
    main()    
        