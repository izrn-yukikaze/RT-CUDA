#ifndef Vec3H__
#define Vec3H__

#define PI 3.1415926535

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <curand_uniform.h>


class Vec3 {
public:
    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(float e0, float e1, float e2) { e[0] = e0, e[1] = e1, e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }

    __host__ __device__ inline Vec3& operator+=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator-=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator/=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const float t);
    __host__ __device__ inline Vec3& operator/=(const float t);

    __host__ __device__ inline float l2() const {
        return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    }

    __host__ __device__ inline float l1() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ inline void unitv();

private:
    float e[3];
};

inline std::istream& operator>>(std::istream &is, Vec3 &t) {
    is >> t[0] >> t[1] >> t[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, Vec3 &t) {
    os << t[0] << " " << t[1] << " " << t[2];
    return os;
}

__host__ __device__ inline void Vec3::unitv() {
    float k = 1.0 / (*this).l2();
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(t * v[0], t * v[1], t * v[2]);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
    return Vec3(v[0] / t, v[1] / t, v[2] / t);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
    return Vec3(t * v[0], t * v[1], t * v[2]);
}

__host__ __device__ inline float dot(const Vec3 &v1, const Vec3 &v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(
        (v1[1] * v2[2] - v1[2] * v2[1]),
        -(v1[0] * v2[2] - v1[2] * v2[0]),
        (v1[0] * v2[1] - v1[1] * v2[0])
    );
}

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3 &v) {
    e[0] += v[0];
    e[1] += v[1];
    e[2] += v[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3 &v) {
    e[0] *= v[0];
    e[1] *= v[1];
    e[2] *= v[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3 &v) {
    e[0] /= v[0];
    e[1] /= v[1];
    e[2] /= v[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
    float k = 1.0/t;
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline Vec3 unitv(Vec3 v) {
    return v / v.l2();
}

__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2 * dot(v, n) * n;
}

#define RANDVEC3 Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vec3 random_in_unit_sphere(curandState *local_rand_state) {
    Vec3 p;
    do {
        p = 2.0f * RANDVEC3 - Vec3(1, 1, 1);
    } while (p.l1() >= 1.0f);
    return p;
}

__host__ __device__ bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) {
    Vec3 uv = unitv(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

class onw {
public:
    __host__ __device__ onw() {}
    __host__ __device__ inline Vec3 operator[](int i) const { return axis[i]; }
    __host__ __device__ Vec3 e1() const { return axis[0]; }
    __host__ __device__ Vec3 e2() const { return axis[1]; }
    __host__ __device__ Vec3 e3() const { return axis[2]; }

    __host__ __device__ Vec3 local(double a, double b, double c){return a*e1() + b*e2() + c*e3();}
    __host__ __device__ Vec3 local(const Vec3& v){return v.x()*e1() + v.y()*e2() + v.z()*e3();}
    __host__ __device__ void build_form_e3(const Vec3& n);


public:
    Vec3 axis[3];
};

__device__ Vec3 random_cosine_direction(curandState *state){

    float r1 = curand_uniform(state);
    float r2 = curand_uniform(state);

    float z = sqrt(1 - r2);

    float phi = 2*PI*r1;
    float x = cos(phi)*sqrt(r2);
    float y = sin(phi)*sqrt(r2);

    return Vec3(x,y,z);
}

__host__ __device__ void onw::build_form_e3(const Vec3& n){
    axis[2] = unitv(n);
    Vec3 a = (fabs(e3().x()) > 0.9 ? Vec3(0,1,0) : Vec3(1,0,0));
    axis[1] = unitv(cross(e3(), a));
    axis[0] = cross(e3(), e2());
}

#endif