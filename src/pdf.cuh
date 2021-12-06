//
// Created by t0hfu on 2021/11/01.
//

#ifndef RT_CUDA_PDF_CUH
#define RT_CUDA_PDF_CUH

class pdf {
public:
    __device__ pdf() {}
    __device__ virtual double value(const Vec3 &direction) const = 0;
    __device__ virtual Vec3 generate(curandState *state) const = 0;
};

__device__ inline Vec3 random_cosine_direction(curandState *state){

    float r1 = curand_uniform(state);
    float r2 = curand_uniform(state);

    float z = sqrt(1 - r2);

    float phi = 2*PI*r1;
    float x = cos(phi)*sqrt(r2);
    float y = sin(phi)*sqrt(r2);

    return Vec3(x,y,z);
}

class cosine_pdf : public pdf {
public:
    __device__ cosine_pdf(const Vec3 &w){ b.build_form_e3(w); }

    __device__ virtual double value(const Vec3 &direction) const {
        double cosine = dot(unitv(direction), b.e3());
        return cosine <= 0 ? 0 : cosine/PI;
    }

    __device__ virtual Vec3 generate(curandState *state) const {
        Vec3 a = random_cosine_direction(state);
        return a.x()*b.e1() + a.y()*b.e2() + a.z()*b.e3();
    }

public:
    onw b;
};

class hittable_pdf : public pdf {
public:
    __device__ hittable_pdf(Entity *p, const Vec3 &origin)
    : ptr(p), o(origin){}

    __device__ virtual double value(const Vec3 &direction) const {
        return ptr->pdf_value(o, direction);
    }

    __device__ virtual Vec3 generate(curandState *state) const {
        return ptr->random(o, state);
    }

public:
    Entity *ptr;
    Vec3 o;
};

class mixture_pdf : public pdf {
public:
    __device__ mixture_pdf(pdf *p0, pdf *p1) {p[0] = p0; p[1] = p1;}

    __device__ virtual double value(const Vec3 &direction) const {
        return 0.5*p[0]->value(direction) + 0.5*p[1]->value(direction);
    }

    __device__ virtual Vec3 generate(curandState *state) const {
        if(curand_uniform(state) < 0.50) return p[0]->generate(state);
        else return p[1]->generate(state);
    }

public:
    pdf *p[2];
};

#endif //RT_CUDA_PDF_CUH
