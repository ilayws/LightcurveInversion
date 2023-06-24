import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import pyopencl as cl
import os

# Global vars
S, n = 0,0
ctx, prg = 0,0


# ---------------------------------------------

data = np.array([])
DISPLAY = True

# Lightcurve settings :
res_x = 600
n_files = 5
n_samples = 1
n_rots = 100
scaler = 0.05
a = 2*np.pi/n_rots

# ---------------------------------------------

# Initialize data to pass to GPU
def generate_render_data(k):
    global S, n
    nsurf = np.int32(S.size//9)
    campos = np.array([30.,0.,0.]).astype(np.float32)
    lightdir = np.array([1,-1,-1]).astype(np.float32)
    lightdir /= np.linalg.norm(lightdir)
    n = n.astype(np.float32)
    S = S.astype(np.float32).reshape(S.size // 3, 3)
    L = np.zeros((k*k), dtype=np.float32)
    raydir = []
    # Ray directions calculation : Using trigonometry
    for x in range(k):
        for y in range(k):
            xx = (2 * ((x + 0.5) / k) - 1)
            yy = (2*((y + 0.5) / k) - 1)
            dir_ = np.array([-4,yy,xx])
            raydir.append(dir_ / np.linalg.norm(dir_))
    raydir = np.array(raydir).flatten().astype(np.float32)
    return nsurf,campos,lightdir,S,n,raydir,L

# Create memory buffers to communicate between gpu and cpu (READ_ONLY = cpu->gpu, WRITE_ONLY = gpu->cpu)
def generate_render_buffers(nsurf,campos,lightdir,S,n,raydir,L):
    campos_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=campos)
    lightdir_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lightdir)
    s_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    n_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n)
    raydir_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=raydir)

    L_buf = cl.Buffer(ctx, mf.WRITE_ONLY, L.nbytes)
    return nsurf,campos_buf,lightdir_buf,s_buf,n_buf,raydir_buf,L_buf

# ---------------------------------------------

# Rotate mesh of shape (ntriangles, 3)
def rotate(mesh, axis, theta):
    q = np.concatenate((axis * np.sin(theta/2), np.array([np.cos(theta/2)])))
    R = Rotation.from_quat(q)
    return R.apply(mesh)

# ---------------------------------------------

# Read obj file to create mesh
def parseObj(fname):
    with open(fname, "r") as file:
      lines = file.readlines()
      v = [np.array(list(map(float, line[1:].split()))) for line in lines if line[0] == "v"]
      f = [np.array(list(map(float, line[1:].split()))) for line in lines if line[0] == "f"]
    mesh = np.array(v)[np.array(f).astype('int') - 1]
    return mesh

# ---------------------------------------------

def run():
    global S, n, data
    for mesh_index in range(n_files):
        fname = "Database/mesh" + str(mesh_index) + ".obj" # <- modify to change shape files to be rendered
        # Initialising variables
        S = parseObj(fname) * scaler # Create mesh array
        n = np.cross(S[:,1]-S[:,0], S[:,2]-S[:,1], axis=1) # Calculate surface normals
        n /= np.linalg.norm(n, axis=1)[:,np.newaxis] # Normalize normal vectors

        nsurf,campos,lightdir,S,n,raydir,L = generate_render_data(res_x)
        nsurf,campos_buf,lightdir_buf,s_buf,n_buf,raydir_buf,L_buf = generate_render_buffers(nsurf,campos,lightdir,S,n,raydir,L)

        # Generate lightcurves and save to file
        for j in range(n_samples):
            light = []
            angle = []
            axis = np.random.rand(3)-0.5
            axis /= np.linalg.norm(axis)
            for k in range(n_rots):
                angle.append((k+1)*a)
                S = rotate(S, axis, a).astype(np.float32)
                n = rotate(n, axis, a).astype(np.float32)
                # Create memory buffers for gpu-cpu communication
                nsurf,campos_buf,lightdir_buf,s_buf,n_buf,raydir_buf,L_buf = generate_render_buffers(nsurf,campos,lightdir,S,n,raydir,L)
                # Run the render() function with L.size (resolution) number of gpu threads
                prg._render(queue, L.shape, None, nsurf,campos_buf,lightdir_buf,s_buf,n_buf,raydir_buf,L_buf)
                # Read result of render from gpu and add to list
                cl.enqueue_copy(queue, L, L_buf)
                light.append(np.sum( L ) / (res_x*res_x))
                # Update maptlotlib display
                if DISPLAY:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    img.set_data(L.reshape(res_x,res_x))
            # Save lightcurve to file in addition to other system data (camera position, etc.)
            dataInstance = np.concatenate((np.array([mesh_index]),axis,np.array([0,0,0]),campos,lightdir,light))[:,np.newaxis]
            if len(data.shape) == 1:
                data = dataInstance.copy()
            else:
                data = np.concatenate((data, dataInstance), axis=1)
                print("Sample : #" + str(data.shape[1]))
                np.savetxt("data.csv", data.T, delimiter=",")


if __name__ == '__main__':

    # Setup GPU
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    os.environ['PYOPOENCL_CTX'] = '0:1'

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Create renderer gpu program
    with open("gpu.txt") as f:
        program = f.read()
    prg = cl.Program(ctx, program).build()

    # Matplotlib displaying the rendered image :
    if DISPLAY:
        plt.ion()
        fig = plt.figure()
        ax = fig.gca()
        img = ax.imshow( np.random.rand(res_x,res_x), cmap="inferno")

    run()