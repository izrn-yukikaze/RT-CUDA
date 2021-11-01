#ifndef METALH__
#define METALH__

#include "material.cuh"
#include "ray.cuh"

class Metal : public Material {
public:
    __device__ Metal(const Vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state, double &pdf) const {
        Vec3 reflected = reflect(unitv(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        pdf = 1.0;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    __device__ virtual double scattering_pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const { return 1.0;}

    Vec3 albedo;
    float fuzz;
};

#endif