#ifndef DIFFUSELIGHTH__
#define DIFFUSELIGHTH__

#include "material.cuh"
#include "texture.cuh"

class DiffuseLight : public Material {
public:
    __device__ DiffuseLight(Texture* tex) : emit(tex) {}
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state, double &pdf) const {
        return false;
    }

    __device__ virtual double scattering_pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const { return 0.0; }

    __device__ Vec3 emitted(float u, float v, const Vec3& p) const {
        return emit->value(u, v, p);
    }

    Texture* emit;
};

#endif