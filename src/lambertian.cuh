#ifndef LAMBERTIANH__
#define LAMBERTIANH__

#include "material.cuh"
#include "vec3.cuh"
#include "entity.cuh"
#include "texture.cuh"


class Lambertian : public Material {
public:
    __device__ Lambertian(Texture* a) : albedo(a) {}
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state, double &pdf) const {
        onw e_base;
        e_base.build_form_e3(rec.normal);
        Vec3 direction = e_base.local(random_cosine_direction(local_rand_state));
        scattered = Ray(rec.p, unitv(direction), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        pdf = dot(e_base.e3(), scattered.direction()) / PI;
        return true;
    }

    __device__ virtual double scattering_pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const {
        auto cosine = dot(rec.normal, unitv(scattered.direction()));
        return cosine < 0 ? 0 : cosine / PI;
    }

    Texture* albedo;
};


#endif