#ifndef MATERIALH__
#define MATERIALH__

#include "entity.cuh"

class Material {
public:
    __device__ virtual Vec3 emitted(float u, float v, const Vec3& p) const { return Vec3(0,0,0); }
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state, double &pdf) const = 0;
    __device__ virtual double scattering_pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const = 0;
};

#endif