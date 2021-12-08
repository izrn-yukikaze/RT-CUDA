cmake_minimum_required(VERSION 3.20)
project(RT_CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_FRAGS -arch=compute_86 -code=sm_86)


set(SOURCE_FILES main.cu)
set(HEADER_FILES
        aabb.cuh
        box.cuh
        bvh_node.cuh
        camera.cuh
        constant_medium.cuh
        diffuse_light.cuh
        entity.cuh
        entitylist.cuh
        isotropic.cuh
        lambertian.cuh
        material.cuh
        metal.cuh
        moving_sphere.cuh
        perlin.cuh
        ray.cuh
        rect.cuh
        sphere.cuh
        stb_image.h
        stb_image_write.h
        texture.cuh
        transform.cuh
        transparent.cuh
        vec3.cuh
        pdf.cuh)

cuda_add_executable(RT_CUDA ${SOURCE_FILES} ${HEADER_FILES})
