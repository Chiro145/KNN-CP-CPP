#include <bits/stdc++.h>
#include <immintrin.h>
#include <cstdint>
#include <omp.h>

using namespace std;

template<typename T,typename...A>void print(const T&t,const A&...a){
std::cout<<t;((std::cout<<' '<<a),...);std::cout<<'\n';}
template<typename...A>void fprint(A&&...a){(cout<<...<<a);}
template<typename...A>void scan(A&...a){((cin>>a),...);}
#define FOR(x, a, b) for (int x = (a); (x) <= (b); ++(x))
#define FOD(x, a, b) for (int x = (a); (x) >= (b); --(x))
#define FOE(x, a)    for (auto &x : (a))
#define ll long long
#define Y  second
#define X  first

const ll  MOD  = 1e9 + 7;
const int MAXN = 2e5 + 7;
const string TRAING_SAMPLE_PATH = "train.csv";
const string TEST_SAMPLE_PATH   = "test.csv";
const int    DIMENSION_SIZE     = 784;
const double TRAIN_SPLIT_RATIO = 0.8;

struct Sample {
    uint8_t pixel[DIMENSION_SIZE];
    uint8_t label;
};

#include "load_csv.h"

vector<Sample> train_samples;
vector<Sample> test_samples;
int train_val_split_idx;
int num_train_samples;
int num_test_samples;
int K = 19;

__attribute__((target("avx512f,avx512bw,avx512dq")))
inline int euclid_distance(const Sample &a, const Sample &b) {
    const uint8_t* p1 = a.pixel;
    const uint8_t* p2 = b.pixel;

    __m512i acc = _mm512_setzero_si512();

    for (int i = 0; i < 768; i += 64) {
        __m512i v1 = _mm512_loadu_si512((const void*)(p1 + i));
        __m512i v2 = _mm512_loadu_si512((const void*)(p2 + i));

        __m256i v1_lo256 = _mm512_castsi512_si256(v1);
        __m256i v1_hi256 = _mm512_extracti64x4_epi64(v1, 1);
        __m256i v2_lo256 = _mm512_castsi512_si256(v2);
        __m256i v2_hi256 = _mm512_extracti64x4_epi64(v2, 1);

        __m512i v1_16_lo = _mm512_cvtepu8_epi16(v1_lo256);
        __m512i v1_16_hi = _mm512_cvtepu8_epi16(v1_hi256);
        __m512i v2_16_lo = _mm512_cvtepu8_epi16(v2_lo256);
        __m512i v2_16_hi = _mm512_cvtepu8_epi16(v2_hi256);

        __m512i diff_lo = _mm512_sub_epi16(v1_16_lo, v2_16_lo);
        __m512i diff_hi = _mm512_sub_epi16(v1_16_hi, v2_16_hi);

        __m512i prod_lo = _mm512_madd_epi16(diff_lo, diff_lo);
        __m512i prod_hi = _mm512_madd_epi16(diff_hi, diff_hi);

        acc = _mm512_add_epi32(acc, prod_lo);
        acc = _mm512_add_epi32(acc, prod_hi);
    }

    int total = _mm512_reduce_add_epi32(acc);

    #pragma GCC unroll 16
    for (int i = 768; i < 784; ++i) {
        int d = (int)p1[i] - (int)p2[i];
        total += d * d;
    }

    return total;
}

inline array<double, 10> query(const Sample &a){
    array<double, 10> prob = {0.0};
    priority_queue<pair<int, uint8_t>> Queue;
    FOR(i, 0, train_val_split_idx - 1){
        Queue.push({euclid_distance(a, train_samples[i]), train_samples[i].label});
        if ((int)Queue.size() > K) Queue.pop();
    } 
    while (!Queue.empty()){
        ++prob[Queue.top().second];
        Queue.pop();
    }
    FOR(i, 0, 9) prob[i] /= K;
    return prob;
}

uint8_t get_major(const array<double, 10> &a){
    double max_prob = 0;
    int major = 0;
    FOR(i, 0, 9)
        if (max_prob < a[i]){
            max_prob = a[i];
            major = i;
        }
    return major;
}

inline double loss(){
    double ret = 0;
    int num_val_samples = num_train_samples - train_val_split_idx;
    const double eps = 1e-15;

#pragma omp parallel for reduction(+:ret) schedule(dynamic)
    for (int i = train_val_split_idx; i < num_train_samples; ++i) {
        double tmp = query(train_samples[i])[train_samples[i].label];
        if (tmp < eps) tmp = eps;
        ret -= log(tmp);
    }
    return ret / num_val_samples;
}

int find_K(int l, int r) {
    l += !(l & 1);

    int sweet_K = l;
    double min_loss = 1e18;

    for (int cur_k = l; cur_k <= r; cur_k += 2) {
        K = cur_k;
        double cur_loss = loss();

        fprint("Testing K = ", cur_k, " | Val Loss: ", fixed, setprecision(5), cur_loss);

        if (cur_loss < min_loss) {
            min_loss = cur_loss;
            sweet_K = cur_k;
            fprint("  <-- [NEW BEST]");
        }
        print("");
    }

    print("=> Sweet K found:", sweet_K, "with Min Loss:", min_loss);
    return sweet_K;
}

int32_t main(void){
    load_train_kaggle(TRAING_SAMPLE_PATH, train_samples);
    load_test_kaggle(TEST_SAMPLE_PATH, test_samples);
    num_train_samples = train_samples.size();
    num_test_samples = test_samples.size();
    train_val_split_idx = num_train_samples * TRAIN_SPLIT_RATIO;
    num_test_samples *= 0.01;
    K = find_K(1, 30);
    print(K);
    FOR(i, 0, num_test_samples - 1)
        print(i, ":", (int)get_major(query(test_samples[i])));
    return 0;
}
