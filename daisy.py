# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:34:12 2018

@author: Rehan
"""

import torch
from scipy import pi
from skimage import data
from skimage import img_as_float
from scipy.ndimage import gaussian_filter
import numpy as np
from numpy import sin, cos
import time
from skimage import img_as_float, draw
from skimage.color import gray2rgb
import matplotlib.pylab as plt
import pdb

def torch_gaussian_filter(imgT, sigma=3, cuda=False):
    # kernelsize is computed as: kernelsize = int(truncate * sigma + 0.5)
    # where default value of truncate is 4.0 as used in gaussian_filter from scipy.
    # Set these to whatever you want for your gaussian filter
    truncate = 4.0
    kernelsize = int(truncate*sigma + 0.5)-1
    kernel_size = kernelsize    
    sigma = sigma
    lim = len(imgT.shape)
    if lim==2:
        channels = 1
    else:
        channels = 3
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).cuda()  if cuda else torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussianfilter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=0)

    gaussianfilter.weight.data = gaussian_kernel
    gaussianfilter.weight.requires_grad = False
    
    # Image padding #
    padding = int((kernel_size-1)/2)
    imagepad = torch.zeros((imgT.shape[0]+padding*2, imgT.shape[1]+padding*2)).cuda() \
        if cuda else torch.zeros((imgT.shape[0]+padding*2, imgT.shape[1]+padding*2))    
    # Copy rows up and down by replicating.
    imagepad[0:padding,padding:imgT.shape[1]+padding] = imgT[0:padding,:]
    imagepad[-padding::,padding:imgT.shape[1]+padding] = imgT[-padding::,:]
    # Copy colum right and left by replicating.
    imagepad[padding:imgT.shape[0]+padding,0:padding] = imgT[:,0:padding]
    imagepad[padding:imgT.shape[0]+padding,-padding::] = imgT[:,-padding::]
    # Copy rest of the image.
    imagepad[padding:-padding,padding:-padding] = imgT
    # Fill the corners...
    # Top left corner.
    imagepad[0:padding,0:padding] = imgT[0:padding,0:padding]
    # Top right corner.
    imagepad[0:padding,-padding::] = imgT[0:padding,-padding::]
    # Bottom left corner
    imagepad[-padding::,0:padding] = imgT[-padding::,0:padding]
    # Bottom right corner
    imagepad[-padding::,-padding::] = imgT[-padding::,-padding::]
    
    if lim==2:
        imagepad = imagepad[np.newaxis,np.newaxis,:,:]
    else:
        imt = torch.zeros(1,3,imagepad.shape[0],imagepad.shape[1]).cuda() if cuda else \
            torch.zeros(1,3,imagepad.shape[0],imagepad.shape[1])
        imt[:,0,:,:] = imgT[:,:,0]
        imt[:,1,:,:] = imgT[:,:,1]
        imt[:,2,:,:] = imgT[:,:,2]
        imagepad = imt

    Imout = gaussianfilter(torch.autograd.Variable(imagepad).float())
    if lim==2:
        Imageout = Imout.data[0,0,:,:]
    else:
        Imageout = Imout.data[0,:,:,:].permute(1,2,0)
    
    return Imageout

def diff_axis_0(a):
    return a[1:]-a[:-1]

def diff_axis_1(a):
    return a[:,1:]-a[:,:-1]

def daisyTorch(image, step=4, radius=15, rings=3, histograms=8, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None, cuda=False, visualize=False):
    imgT = torch.from_numpy(image).float()
    # load image into GPU if cuda flag is True.
    if cuda==True:
        torch.cuda.init()
        imgT = imgT.cuda()

    # Validate parameters.
    if sigmas is not None and ring_radii is not None \
            and len(sigmas) - 1 != len(ring_radii):
        raise ValueError('`len(sigmas)-1 != len(ring_radii)`')
    if ring_radii is not None:
        rings = len(ring_radii)
        radius = ring_radii[-1]
    if sigmas is not None:
        rings = len(sigmas) - 1
    if sigmas is None:
        sigmas = [radius * (i + 1) / float(2 * rings) for i in range(rings)]
    if ring_radii is None:
        ring_radii = [radius * (i + 1) / float(rings) for i in range(rings)]
    if normalization not in ['l1', 'l2', 'daisy', 'off']:
        raise ValueError('Invalid normalization method.')
    
    # Compute image derivatives.
    dx = torch.zeros(imgT.shape).cuda() if cuda else torch.zeros(imgT.shape)
    dy = torch.zeros(imgT.shape).cuda() if cuda else torch.zeros(imgT.shape)
    dx[:,:-1] = diff_axis_1(imgT)
    dy[:-1,:] = diff_axis_0(imgT)
    
    # Compute gradient orientation and magnitude and their contribution
    # to the histograms.
    grad_mag = torch.sqrt(dx ** 2 + dy ** 2)
    grad_ori = torch.atan2(dy, dx)
    orientation_kappa = orientations / pi
    orientation_angles = [2 * o * pi / orientations - pi
                          for o in range(orientations)]
    
    hist = torch.zeros((orientations,) + image.shape).cuda() if cuda else torch.zeros((orientations,) + image.shape) 
    for i, o in enumerate(orientation_angles):
        # Weigh bin contribution by the circular normal distribution
        hist[i, :, :] = torch.exp(orientation_kappa * torch.cos(grad_ori - o))
        # Weigh bin contribution by the gradient magnitude
        hist[i, :, :] = torch.mul(hist[i, :, :], grad_mag)
    
    # Smooth orientation histograms for the center and all rings.
    sigmas = [sigmas[0]] + sigmas
    hist_smooth = torch.zeros((rings + 1,) + hist.shape).cuda() if cuda else torch.zeros((rings + 1,) + hist.shape)
  
    for i in range(rings + 1):
        for j in range(orientations):
            #hist_smooth[i, j, :, :] = torch.from_numpy(gaussian_filter(hist[j, :, :].cpu().numpy(),sigma=sigmas[i]))
            hist_smooth[i, j, :, :] = torch_gaussian_filter(hist[j, :, :],sigma=sigmas[i], cuda=cuda)
    
    # Assemble descriptor grid.
    theta = [2 * pi * j / histograms for j in range(histograms)]
    desc_dims = (rings * histograms + 1) * orientations
    descs = torch.zeros((desc_dims, image.shape[0]-2*radius, image.shape[1]-2*radius)).cuda() if cuda \
        else torch.zeros((desc_dims, image.shape[0]-2*radius, image.shape[1]-2*radius))
    descs[:orientations, :, :] = hist_smooth[0, :, radius:-radius, radius:-radius]
    
    idx = orientations
    for i in range(rings):
        for j in range(histograms):
            y_min = radius + int(round(ring_radii[i] * sin(theta[j])))
            y_max = descs.shape[1] + y_min
            x_min = radius + int(round(ring_radii[i] * cos(theta[j])))
            x_max = descs.shape[2] + x_min
            descs[idx:idx + orientations, :, :] = hist_smooth[i + 1, :,
                                                              y_min:y_max,
                                                              x_min:x_max]
            idx += orientations
    descs = descs[:, ::step, ::step]
    descs = descs.permute(1,2,0)
    
    # Normalize descriptors.
    if normalization != 'off':
        descs += 1e-10
        if normalization == 'l1':
            descs /= torch.sum(descs, dim=2)[:, :, np.newaxis]
        elif normalization == 'l2':
            descs /= torch.sqrt(torch.sum(descs ** 2, dim=2))[:, :, np.newaxis]
        elif normalization == 'daisy':
            for i in range(0, desc_dims, orientations):
                norms = torch.sqrt(torch.sum(descs[:, :, i:i + orientations] ** 2, dim=2))
                descs[:, :, i:i + orientations] /= norms[:, :, np.newaxis]

    if visualize:
        np_descs = descs.cpu().numpy()
        img = img_as_float(image)

        descs_img = gray2rgb(img)
        for i in range(np_descs.shape[0]):
            for j in range(np_descs.shape[1]):
                # Draw center histogram sigma
                color = [1, 0, 0]
                desc_y = i * step + radius
                desc_x = j * step + radius
                rows, cols, val = draw.circle_perimeter_aa(desc_y, desc_x, int(sigmas[0]))
                draw.set_color(descs_img, (rows, cols), color, alpha=val)
                max_bin = np.max(np_descs[i, j, :])
                for o_num, o in enumerate(orientation_angles):
                    # Draw center histogram bins
                    bin_size = np_descs[i, j, o_num] / max_bin
                    dy = sigmas[0] * bin_size * sin(o)
                    dx = sigmas[0] * bin_size * cos(o)
                    rows, cols, val = draw.line_aa(desc_y, desc_x, int(desc_y + dy),
                                                   int(desc_x + dx))
                    draw.set_color(descs_img, (rows, cols), color, alpha=val)
                for r_num, r in enumerate(ring_radii):
                    color_offset = float(1 + r_num) / rings
                    color = (1 - color_offset, 1, color_offset)
                    for t_num, t in enumerate(theta):
                        # Draw ring histogram sigmas
                        hist_y = desc_y + int(round(r * sin(t)))
                        hist_x = desc_x + int(round(r * cos(t)))
                        rows, cols, val = draw.circle_perimeter_aa(hist_y, hist_x,
                                                                   int(sigmas[r_num + 1]))
                        draw.set_color(descs_img, (rows, cols), color, alpha=val)
                        for o_num, o in enumerate(orientation_angles):
                            # Draw histogram bins
                            bin_size = np_descs[i, j, orientations + r_num *
                                             histograms * orientations +
                                             t_num * orientations + o_num]
                            bin_size /= max_bin
                            dy = sigmas[r_num + 1] * bin_size * sin(o)
                            dx = sigmas[r_num + 1] * bin_size * cos(o)
                            rows, cols, val = draw.line_aa(hist_y, hist_x,
                                                           int(hist_y + dy),
                                                           int(hist_x + dx))
                            draw.set_color(descs_img, (rows, cols), color, alpha=val)
        return descs, descs_img
    
    return descs

if __name__ == '__main__':
    img = data.camera()
    image = img_as_float(img)
    tic = time.time()
        
    descriptor = daisyTorch(image=image, step=180, radius=58, rings=2, 
                            histograms=6, orientations=8, normalization='l1',cuda=True)

    toc = time.time()    
    print("Total time: %.4f sec" %(toc-tic))
    