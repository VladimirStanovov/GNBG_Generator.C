/// Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
/// Last edited: March 10th, 2024
/// C++ implementation of Generalized Numerical Benchmark Generator (GNBG)
/// Includes implementation of simple Differential Evolution (DE) with rand/1 strategy and binomial crossover
/// Problem parameters can be saved to file for further usage
/// Competition page: https://competition-hub.github.io/GNBG-Competition/
/// Reference:
/// D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb, and A. H. Gandomi, "GNBG: A Generalized
///   and Configurable Benchmark Generator for Continuous Numerical Optimization," arXiv prepring	arXiv:2312.07083, 2023.
/// A. H. Gandomi, D. Yazdani, M. N. Omidvar, and K. Deb, "GNBG-Generated Test Suite for Box-Constrained Numerical Global
///   Optimization," arXiv preprint arXiv:2312.07034, 2023.
/// MATLAB version: https://github.com/Danial-Yazdani/GNBG_Generator.MATLAB
/// Python version: https://github.com/Danial-Yazdani/GNBG_Generator.Python
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <random>
#include <cstring> // for memset
#define PI 3.1415926535897932384626433832795029
using namespace std;
unsigned globalseed = unsigned(time(NULL));//2024;//
std::mt19937 generator_uni_i(globalseed);
std::mt19937 generator_uni_r(globalseed+100);
std::mt19937 generator_norm(globalseed+200);
std::mt19937 generator_beta(globalseed+300);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
std::normal_distribution<double> norm_dist(0.0,1.0);
int IntRandom(int target)
{
    if(target == 0) return 0;
    return uni_int(generator_uni_i)%target;
}
double Random(double minimal, double maximal)
{
    return uni_real(generator_uni_r)*(maximal-minimal)+minimal;
}
double NormRand(double mu, double sigma)
{
    return norm_dist(generator_norm)*sigma + mu;
}
const double bd_alpha = 0.2; //beta distribution alpha
const double bd_beta =  0.2; //beta distribution beta

/* Beta distribution implementation for C++ taken from here: https://stackoverflow.com/questions/15165202/random-number-generator-with-beta-distribution */
template <typename RealType = double>
class beta_distribution
{
public:
    typedef RealType result_type;

    class param_type
    {
    public:
        typedef beta_distribution distribution_type;

        explicit param_type(RealType a = 2.0, RealType b = 2.0)
            : a_param(a), b_param(b) { }

        RealType a() const
        {
            return a_param;
        }
        RealType b() const
        {
            return b_param;
        }

        bool operator==(const param_type& other) const
        {
            return (a_param == other.a_param &&
                    b_param == other.b_param);
        }

        bool operator!=(const param_type& other) const
        {
            return !(*this == other);
        }

    private:
        RealType a_param, b_param;
    };

    explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
        : a_gamma(a), b_gamma(b) { }
    explicit beta_distribution(const param_type& param)
        : a_gamma(param.a()), b_gamma(param.b()) { }

    void reset() { }

    param_type param() const
    {
        return param_type(a(), b());
    }

    void param(const param_type& param)
    {
        a_gamma = gamma_dist_type(param.a());
        b_gamma = gamma_dist_type(param.b());
    }

    template <typename URNG>
    result_type operator()(URNG& engine)
    {
        return generate(engine, a_gamma, b_gamma);
    }

    template <typename URNG>
    result_type operator()(URNG& engine, const param_type& param)
    {
        gamma_dist_type a_param_gamma(param.a()),
                        b_param_gamma(param.b());
        return generate(engine, a_param_gamma, b_param_gamma);
    }

    result_type min() const
    {
        return 0.0;
    }
    result_type max() const
    {
        return 1.0;
    }

    result_type a() const
    {
        return a_gamma.alpha();
    }
    result_type b() const
    {
        return b_gamma.alpha();
    }

    bool operator==(const beta_distribution<result_type>& other) const
    {
        return (param() == other.param() &&
                a_gamma == other.a_gamma &&
                b_gamma == other.b_gamma);
    }

    bool operator!=(const beta_distribution<result_type>& other) const
    {
        return !(*this == other);
    }

private:
    typedef std::gamma_distribution<result_type> gamma_dist_type;

    gamma_dist_type a_gamma, b_gamma;

    template <typename URNG>
    result_type generate(URNG& engine,
                         gamma_dist_type& x_gamma,
                         gamma_dist_type& y_gamma)
    {
        result_type x = x_gamma(engine);
        return x / (x + y_gamma(engine));
    }
};

template <typename CharT, typename RealType>
std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT>& os,
                                      const beta_distribution<RealType>& beta)
{
    os << "~Beta(" << beta.a() << "," << beta.b() << ")";
    return os;
}

template <typename CharT, typename RealType>
std::basic_istream<CharT>& operator>>(std::basic_istream<CharT>& is,
                                      beta_distribution<RealType>& beta)
{
    std::string str;
    RealType a, b;
    if (std::getline(is, str, '(') && str == "~Beta" &&
            is >> a && is.get() == ',' && is >> b && is.get() == ')')
    {
        beta = beta_distribution<RealType>(a, b);
    }
    else
    {
        is.setstate(std::ios::failbit);
    }
    return is;
}
beta_distribution<> beta(bd_alpha, bd_beta);
double BetaRand() {return beta(generator_beta);}

void matmul3(double** matrix1, double** matrix2, double** matrix3, const int dim)
{
    for(int i=0;i!=dim;i++)
    {
        for(int j=0;j!=dim;j++)
        {
            matrix3[i][j] = 0;
            for(int k=0;k!=dim;k++)
                matrix3[i][j] += matrix1[i][k]*matrix2[k][j];
        }
    }
}

class GNBG
{
public:
    int FEval;
    int MaxEvals;
    int Dimension;
    int CompNum;
    int ComponentPositioningMethod;
    int SigmaPattern;
    int H_pattern;
    int localModalitySymmetry;
    int Rotation;
    int LambdaConfigMethod;
    int OptimumIndex;
    double MinCoordinate;
    double MaxCoordinate;
    double AcceptanceThreshold;
    double OptimumValue;
    double BestFoundResult;
    double fval;
    double AcceptanceReachPoint;
    double MinRandOptimaPos;
    double MaxRandOptimaPos;
    double MinExclusiveRange;
    double MaxExclusiveRange;
    double MinSigma;
    double MaxSigma;
    double MinMu;
    double MaxMu;
    double MinOmega;
    double MaxOmega;
    double MinAngle;
    double MaxAngle;
    double MinLambda;
    double MaxLambda;
    double LambdaValue4ALL;
    double* CompSigma;
    double* Lambda;
    double* OptimumPosition;
    double* FEhistory;
    double* temp;
    double* a;
    double** CompMinPos;
    double** CompH;
    double** Mu;
    double** Omega;
    double*** RotationMatrix;
    double** ThetaMatrix;
    double** temp2;
    double** temp3;
    GNBG(){};
    void Init();
    void Clear();
    ~GNBG(){};
    void initialize_component_position();
    void initialize_component_sigma();
    void initialize_component_H();
    void initialize_modality_parameters();
    void initialize_lambda();
    void define_rotation_matrices();
    void rotation(double** theta, double** RotationMatrixPart, double** temp2, double** temp3);
    void transform_(double* X, double* alpha, double* beta);
    void save_to_file(string filename);
    double Fitness(double* xvec);
};
void GNBG::Init()
{
    MaxEvals = 100000;
    AcceptanceThreshold = 1e-8;
    Dimension = 2;
    CompNum = 3;
    MinCoordinate = -100;
    MaxCoordinate = 100;
    FEval = 0;
    AcceptanceReachPoint = -1;
    FEhistory = new double[MaxEvals];
    a = new double[Dimension];
    temp = new double[Dimension];
    OptimumPosition = new double[Dimension];
    ThetaMatrix = new double*[Dimension];
    temp2 = new double*[Dimension];
    temp3 = new double*[Dimension];
    for(int i=0;i!=Dimension;i++)
    {
        ThetaMatrix[i] = new double[Dimension];
        temp2[i] = new double[Dimension];
        temp3[i] = new double[Dimension];
    }
    CompSigma = new double[CompNum];
    Lambda = new double[CompNum];
    CompMinPos = new double*[CompNum];
    CompH = new double*[CompNum];
    Mu = new double*[CompNum];
    Omega = new double*[CompNum];
    RotationMatrix = new double**[CompNum];
    for(int i=0;i!=CompNum;i++)
    {
        CompMinPos[i] = new double[Dimension];
        CompH[i] = new double[Dimension];
        Mu[i] = new double[2];
        Omega[i] = new double[4];
        RotationMatrix[i] = new double*[Dimension];
        for(int j=0;j!=Dimension;j++)
            RotationMatrix[i][j] = new double[Dimension];
    }
    initialize_component_position();
    initialize_component_sigma();
    initialize_component_H();
    initialize_modality_parameters();
    initialize_lambda();
    define_rotation_matrices();
    OptimumValue = CompSigma[0];
    OptimumIndex = 0;
    for(int i=0;i!=CompNum;i++)
    {
        if(CompSigma[i] < OptimumValue)
        {
            OptimumValue = CompSigma[i];
            OptimumIndex = i;
        }
    }
    for(int i=0;i!=Dimension;i++)
        OptimumPosition[i] = CompMinPos[OptimumIndex][i];
}
void GNBG::initialize_component_position()
{
    MinRandOptimaPos = -80;
    MaxRandOptimaPos =  80;
    MinExclusiveRange = -30; //Must be LARGER than MinRandOptimaPos
    MaxExclusiveRange =  30; //Must be SMALLER than MaxRandOptimaPos
    ComponentPositioningMethod = 1;
    /*(1) Random positions with uniform distribution inside the search range
      (2) Random positions with uniform distribution inside a specified range [GNBG.MinRandOptimaPos,GNBG.MaxRandOptimaPos]
      (3) Random positions inside a specified range [GNBG.MinRandOptimaPos,GNBG.MaxRandOptimaPos] but not within the sub-range [GNBG.MinExclusiveRange,GNBG.MaxExclusiveRange]
      (4) Random OVERLAPPING positions with uniform distribution inside a specified range [GNBG.MinRandOptimaPos,GNBG.MaxRandOptimaPos]. Remember to also set GNBG.SigmaPattern to 2.*/
    if(ComponentPositioningMethod == 1)
    {
        for(int j=0;j!=CompNum;j++)
            for(int k=0;k!=Dimension;k++)
                CompMinPos[j][k] = Random(MinCoordinate,MaxCoordinate);
    }
    else if(ComponentPositioningMethod == 2)
    {
        for(int j=0;j!=CompNum;j++)
            for(int k=0;k!=Dimension;k++)
                CompMinPos[j][k] = Random(MinRandOptimaPos,MaxRandOptimaPos);
    }
    else if(ComponentPositioningMethod == 3)
    {
        for(int j=0;j!=CompNum;j++)
            for(int k=0;k!=Dimension;k++)
            {
                bool selector = IntRandom(2);
                CompMinPos[j][k] = selector*(MinRandOptimaPos + Random(0,1)*(MinExclusiveRange - MinRandOptimaPos)) +
                                (1-selector)*(MaxExclusiveRange + Random(0,1)*(MaxRandOptimaPos - MaxExclusiveRange));
            }
    }
    else if(ComponentPositioningMethod == 4)
    {
        for(int k=0;k!=Dimension;k++)
        {
            double val = MinRandOptimaPos + (MaxRandOptimaPos - MinRandOptimaPos)*Random(0,1);
            for(int j=0;j!=CompNum;j++)
                CompMinPos[j][k] = val;
        }
    }
    else
        cout<<"Warning: Wrong number is chosen for ComponentPositioningMethod."<<endl;
}
void GNBG::initialize_component_sigma()
{
    MinSigma = -99;
    MaxSigma = -98;
    SigmaPattern = 1;
    /*(1) A random sigma value for EACH component.
      (2) A random sigma value for ALL components. It must be used for generating overlapping scenarios, or when the user plans to generate problem instances with multiple global optima.
      (3) Manually defined values for sigma.*/
    if(SigmaPattern == 1)
    {
        for(int j=0;j!=CompNum;j++)
            CompSigma[j] = Random(MinSigma,MaxSigma);
    }
    else if(SigmaPattern == 2)
    {
        double val = Random(MinSigma,MaxSigma);
        for(int j=0;j!=CompNum;j++)
            CompSigma[j] = val;
    }
    else if(SigmaPattern == 2)
    {
        //USER-DEFINED ==> Adjust the size of this array to match the number of components
        CompSigma[0] = -1000;
        CompSigma[1] = -950;
    }
    else
        cout<<"Warning: Wrong number is chosen for SigmaPattern."<<endl;
}
void GNBG::initialize_component_H()
{
    int H_pattern = 4;
    /*(1) Condition number is 1 and all elements of principal diagonal of H are set to a user defined value H_value
      (2) Condition number is 1 for all components but the elements of principal diagonal of H are different from a component to another and are randomly generated with uniform distribution within the range [Lb_H, Ub_H].
      (3) Condition number is random for all components the values of principal diagonal of the matrix H for each component are generated randomly within the range [Lb_H, Ub_H] using a uniform distribution.
      (4) Condition number is Ub_H/Lb_H for all components where two randomly selected elements on the principal diagonal of the matrix H are explicitly set to Lb_H and Ub_H. The remaining diagonal elements are generated randomly within the range [Lb_H, Ub_H]. These values follow a Beta distribution characterized by user-defined parameters alpha and beta, where 0 < alpha = beta <= 1.
      (5) Condition number is Ub_H/Lb_H for all components where a vector with Dimension equally spaced values between Lb_H and Ub_H is generated. The linspace function is used to create a linearly spaced vector that includes both the minimum and maximum values. For each component, a randomly permutation of this vector is used.*/
    double Lb_H = 1;
    double Ub_H = 10000;
    if(H_pattern == 1)
    {
        for(int j=0;j!=CompNum;j++)
            for(int k=0;k!=Dimension;k++)
                CompH[j][k] = 1;
    }
    else if(H_pattern == 2)
    {
        for(int k=0;k!=Dimension;k++)
        {
            double val = Random(Lb_H,Ub_H);
            for(int j=0;j!=CompNum;j++)
                CompH[j][k] = val;
        }
    }
    else if(H_pattern == 3)
    {
        for(int j=0;j!=CompNum;j++)
            for(int k=0;k!=Dimension;k++)
                CompH[j][k] = Random(Lb_H,Ub_H);
    }
    else if(H_pattern == 4)
    {
        for(int j=0;j!=CompNum;j++)
            for(int k=0;k!=Dimension;k++)
                CompH[j][k] = BetaRand();
        for(int j=0;j!=CompNum;j++)
        {
            int index1 = IntRandom(Dimension);
            int index2 = IntRandom(Dimension-1);
            index2 = index2 + (index2 >= index1); //so that index1 and index2 are different
            CompH[j][index1] = Lb_H;
            CompH[j][index2] = Ub_H;
        }
    }
    else if(H_pattern == 5)
    {
        int* tmpindex = new int[Dimension];
        for(int k=0;k!=Dimension;k++)
            tmpindex[k] = k;
        for(int j=0;j!=CompNum;j++)
        {
            for(int k=0;k!=Dimension*2;k++)
                swap(tmpindex[IntRandom(Dimension)],tmpindex[IntRandom(Dimension)]);
            for(int k=0;k!=Dimension;k++)
            {
                CompH[j][tmpindex[k]] = Lb_H + 1.0/double(Dimension-1)*(Ub_H - Lb_H)*k;
            }
        }
        delete tmpindex;
    }
    else
        cout<<"Warning: Wrong number is chosen for H_pattern."<<endl;
}
void GNBG::initialize_modality_parameters()
{
    MinMu = 0.2;
    MaxMu = 0.5;
    MinOmega = 5;
    MaxOmega = 50;
    localModalitySymmetry = 3;
    /*(1) Unimodal, smooth, and regular components
      (2) Multimodal symmetric components
      (3) Multimodal asymmetric components
      (4) Manually defined values*/
    if(localModalitySymmetry == 1)
    {
        for(int j=0;j!=CompNum;j++)
        {
            Mu[j][0] = 0;
            Mu[j][1] = 0;
            Omega[j][0] = 0;
            Omega[j][1] = 0;
            Omega[j][2] = 0;
            Omega[j][3] = 0;
        }
    }
    else if(localModalitySymmetry == 2)
    {
        for(int j=0;j!=CompNum;j++)
        {
            Mu[j][0] = Random(MinMu,MaxMu);
            Mu[j][1] = Mu[j][0];
            Omega[j][0] = Random(MinOmega,MaxOmega);
            Omega[j][1] = Omega[j][0];
            Omega[j][2] = Omega[j][0];
            Omega[j][3] = Omega[j][0];
        }
    }
    else if(localModalitySymmetry == 3)
    {
        for(int j=0;j!=CompNum;j++)
        {
            Mu[j][0] = Random(MinMu,MaxMu);
            Mu[j][1] = Random(MinMu,MaxMu);
            Omega[j][0] = Random(MinMu,MaxMu);
            Omega[j][1] = Random(MinMu,MaxMu);
            Omega[j][2] = Random(MinMu,MaxMu);
            Omega[j][3] = Random(MinMu,MaxMu);
        }
    }
    else if(localModalitySymmetry == 4)
    {
        // Assuming self.CompNum is defined and matches the required shape
        // User-defined values; adjust sizes as needed
        for(int j=0;j!=CompNum;j++)
        {
            Mu[j][0] = 1;
            Mu[j][1] = 1;
            Omega[j][0] = 10;
            Omega[j][1] = 10;
            Omega[j][2] = 10;
            Omega[j][3] = 10;
        }
    }
    else
        cout<<"Warning: Wrong number is chosen for localModalitySymmetry."<<endl;
}
void GNBG::initialize_lambda()
{
    MinLambda = 1;
    MaxLambda = 1;
    LambdaValue4ALL = 0.25;
    LambdaConfigMethod = 1;
    /* 1 All lambda are set to LambdaValue4ALL
       2 Randomly set lambda of each component in [MinLambda,MaxLambda]. Note that large ranges may result in existence of invisible components*/
    if(LambdaConfigMethod == 1)
    {
        for(int i=0;i!=CompNum;i++)
            Lambda[i] = LambdaValue4ALL;
    }
    else if(LambdaConfigMethod == 2)
    {
        for(int i=0;i!=CompNum;i++)
            Lambda[i] = Random(MinLambda,MaxLambda);
    }
    else
        cout<<"Warning: Wrong number is chosen for LambdaConfigMethod."<<endl;
}
void GNBG::define_rotation_matrices()
{
    MinAngle = -PI;
    MaxAngle =  PI;
    Rotation = 2;
    if(Rotation == 1)
    {
        for(int i=0;i!=CompNum;i++)
            for(int j=0;j!=Dimension;j++)
            {
                for(int k=0;k!=Dimension;k++)
                    RotationMatrix[i][j][k] = 0;
                RotationMatrix[i][j][j] = 1;
            }
    }
    else if(Rotation == 2)
    {
        for(int i=0;i!=CompNum;i++)
        {
            for(int j=0;j!=Dimension;j++)
                for(int k=0;k!=Dimension;k++)
                    ThetaMatrix[j][k] = Random(MinAngle,MaxAngle)*(k > j);
            rotation(ThetaMatrix,RotationMatrix[i],temp2,temp3);
        }
    }
    else if(Rotation == 3)
    {
        double MinConProb = 0.3;
        double MaxConProb = 0.75;
        for(int i=0;i!=CompNum;i++)
        {
            double ConnectionProbability = Random(MinConProb,MaxConProb);
            for(int j=0;j!=Dimension;j++)
                for(int k=0;k!=Dimension;k++)
                    ThetaMatrix[j][k] = Random(MinAngle,MaxAngle)*(Random(0,1) < ConnectionProbability);
            rotation(ThetaMatrix,RotationMatrix[i],temp2,temp3);
        }
    }
    else if(Rotation == 4)
    {
        double RandomAngle = Random(MinAngle,MaxAngle);
        for(int i=0;i!=CompNum;i++)
        {
            for(int j=0;j!=Dimension;j++)
                for(int k=0;k!=Dimension;k++)
                    ThetaMatrix[j][k] = RandomAngle*(k > j);
            rotation(ThetaMatrix,RotationMatrix[i],temp2,temp3);
        }
    }
    else if(Rotation == 5)
    {
        double SpecificAngles = 1.48353;
        for(int i=0;i!=CompNum;i++)
        {
            for(int j=0;j!=Dimension;j++)
                for(int k=0;k!=Dimension;k++)
                    ThetaMatrix[j][k] = SpecificAngles*(k > j);
            rotation(ThetaMatrix,RotationMatrix[i],temp2,temp3);
        }
    }
    else if(Rotation == 6)
    {
        for(int i=0;i!=CompNum;i++)
        {
            for(int j=0;j!=Dimension;j++)
                for(int k=0;k!=Dimension;k++)
                    ThetaMatrix[j][k] = 0;
            for(int j=0;j!=Dimension-1;j++)
                ThetaMatrix[j][j+1] = Random(MinAngle,MaxAngle);
            rotation(ThetaMatrix,RotationMatrix[i],temp2,temp3);
        }
    }
    else if(Rotation == 7)
    {
        int NGroups = 3;
        if(NGroups > Dimension)
            cout<<"Warning: The number of groups exceeds the Dimension"<<endl;
        int* Svals = new int[NGroups];
        double* Theta = new double[NGroups];
        Svals[0] = 3;
        Svals[1] = 4;
        Svals[2] = 3;
        Theta[0] = PI/4.0;
        Theta[1] = 3*PI/4.0;
        Theta[2] = PI/8.0;
        int SumS = 0;
        for(int i=0;i!=NGroups;i++)
            SumS += Svals[i];
        if(SumS != NGroups)
            cout<<"Warning: The sum of elements in S is not equal to the Dimension"<<endl;
        int* tmpindex = new int[Dimension];
        for(int k=0;k!=Dimension;k++)
            tmpindex[k] = k;
        for(int i=0;i!=CompNum;i++)
        {
            for(int j=0;j!=Dimension;j++)
                for(int k=0;k!=Dimension;k++)
                    ThetaMatrix[j][k] = 0;
            for(int k=0;k!=Dimension*2;k++)
                swap(tmpindex[IntRandom(Dimension)],tmpindex[IntRandom(Dimension)]);
            int groupStart = 0;
            for(int j=0;j!=NGroups;j++)
            {
                for(int k=0;k!=Svals[j];k++)
                    for(int L=0;L!=Svals[j];L++)
                        if(tmpindex[groupStart+k] < tmpindex[groupStart+L])
                            ThetaMatrix[tmpindex[groupStart+k]][tmpindex[groupStart+L]] = Theta[j];
                groupStart += Svals[j];
            }
            rotation(ThetaMatrix,RotationMatrix[i],temp2,temp3);
        }
        delete Svals;
        delete Theta;
        delete tmpindex;
    }
    else
        cout<<"Wrong number is chosen for Rotation"<<endl;
}
void GNBG::rotation(double** theta, double** RotationMatrixPart, double** temp2, double** temp3)
{
    for(int k=0;k!=Dimension;k++)
    {
        for(int L=0;L!=Dimension;L++)
            RotationMatrixPart[k][L] = 0;
        RotationMatrixPart[k][k] = 1;
    }
    for(int i=0;i!=Dimension-1;i++)
    {
        for(int j=i+1;j!=Dimension;j++)
        {
            if(theta[i][j] != 0)
            {
                for(int k=0;k!=Dimension;k++)
                {
                    for(int L=0;L!=Dimension;L++)
                        temp2[k][L] = 0;
                    temp2[k][k] = 1;
                }
                temp2[i][i] = cos(theta[i][j]);
                temp2[j][j] = temp2[i][i];
                temp2[j][i] = sin(theta[i][j]);
                temp2[i][j] =-temp2[j][i];
                matmul3(RotationMatrixPart,temp2,temp3,Dimension);
                for(int k=0;k!=Dimension;k++)
                    for(int L=0;L!=Dimension;L++)
                        RotationMatrixPart[k][L] = temp3[k][L];
            }
        }
    }
}
double GNBG::Fitness(double* xvec)
{
    double res = 0;
    for(int i=0;i!=CompNum;i++)
    {
        for(int j=0;j!=Dimension;j++)
            a[j] = xvec[j] - CompMinPos[i][j];
        for(int j=0;j!=Dimension;j++)
        {
            temp[j] = 0;
            for(int k=0;k!=Dimension;k++)
                temp[j] += RotationMatrix[i][j][k]*a[k]; //matmul rotation matrix and (x - peak position)
        }
        for(int j=0;j!=Dimension;j++)
        {
            if(temp[j] > 0)
                a[j] = exp(log( temp[j])+Mu[i][0]*(sin(Omega[i][0]*log( temp[j]))+sin(Omega[i][1]*log( temp[j]))));
            else if(temp[j] < 0)
                a[j] =-exp(log(-temp[j])+Mu[i][1]*(sin(Omega[i][2]*log(-temp[j]))+sin(Omega[i][3]*log(-temp[j]))));
            else
                a[j] = 0;
        }
        fval = 0;
        for(int j=0;j!=Dimension;j++)
            fval += a[j]*a[j]*CompH[i][j];
        fval = CompSigma[i] + pow(fval,Lambda[i]);
        //if first iter then save fval, else take min
        res = (i == 0)*fval + (i != 0)*min(res,fval);
    }
    if(FEval > MaxEvals)
        return res;
    FEhistory[FEval] = res;
    BestFoundResult = min(res,BestFoundResult);
    if(FEhistory[FEval] - OptimumValue < AcceptanceThreshold && AcceptanceReachPoint == -1)
       AcceptanceReachPoint = FEval;
    FEval++;
    return res;
}
void GNBG::Clear()
{
    delete a;
    delete temp;
    delete OptimumPosition;
    delete CompSigma;
    delete Lambda;
    for(int i=0;i!=CompNum;i++)
    {
        delete CompMinPos[i];
        delete CompH[i];
        delete Mu[i];
        delete Omega[i];
        for(int j=0;j!=Dimension;j++)
            delete RotationMatrix[i][j];
        delete RotationMatrix[i];
    }
    delete CompMinPos;
    delete CompH;
    delete Mu;
    delete Omega;
    delete RotationMatrix;
    delete FEhistory;
    for(int i=0;i!=Dimension;i++)
    {
        delete ThetaMatrix[i];
        delete temp2[i];
        delete temp3[i];
    }
    delete ThetaMatrix;
    delete temp2;
    delete temp3;
}
void GNBG::save_to_file(string filename)
{
    ofstream fout(filename.c_str());
    fout.precision(20);
    fout<<MaxEvals<<endl;
    fout<<AcceptanceThreshold<<endl;
    fout<<Dimension<<endl;
    fout<<CompNum<<endl;
    fout<<MinCoordinate<<endl;
    fout<<MaxCoordinate<<endl;
    for(int i=0;i!=CompNum;i++)
    {
        for(int j=0;j!=Dimension;j++)
            fout<<CompMinPos[i][j]<<"\t";
        fout<<endl;
    }
    for(int i=0;i!=CompNum;i++)
        fout<<CompSigma[i]<<"\t";
    fout<<endl;
    for(int i=0;i!=CompNum;i++)
    {
        for(int j=0;j!=Dimension;j++)
            fout<<CompH[i][j]<<"\t";
        fout<<endl;
    }
    for(int i=0;i!=CompNum;i++)
    {
        for(int j=0;j!=2;j++)
            fout<<Mu[i][j]<<"\t";
        fout<<endl;
    }
    for(int i=0;i!=CompNum;i++)
    {
        for(int j=0;j!=4;j++)
            fout<<Omega[i][j]<<"\t";
        fout<<endl;
    }
    for(int i=0;i!=CompNum;i++)
        fout<<Lambda[i]<<"\t";
    fout<<endl;
    for(int j=0;j!=Dimension;j++)
        for(int k=0;k!=Dimension;k++)
        {
            for(int i=0;i!=CompNum;i++)
                fout<<RotationMatrix[i][j][k]<<"\t";
            fout<<endl;
        }
    fout<<OptimumValue<<endl;
    for(int i=0;i!=Dimension;i++)
        fout<<OptimumPosition[i]<<"\t";
    fout<<endl;
}

class Optimizer
{
public:
    int NInds;
    int NVars;
    int BestHistoryIndex;
    double F;
    double Cr;
    double Left;
    double Right;
    double BestF;
    double tempfit;
    double** Popul;
    double* Fitness;
    double* Trial;
    double* BestHistory;
    Optimizer(int newNInds, GNBG& gnbg);
    ~Optimizer();
    void Run(GNBG& gnbg);
};
Optimizer::Optimizer(int newNInds, GNBG& gnbg)
{
    NInds = newNInds;
    NVars = gnbg.Dimension;
    Left = gnbg.MinCoordinate;
    Right = gnbg.MaxCoordinate;
    BestHistoryIndex = 0;
    Popul = new double*[NInds];
    Fitness = new double[NInds];
    Trial = new double[NVars];
    BestHistory = new double[int(gnbg.MaxEvals*1.0)];
    for(int i=0;i!=NInds;i++)
    {
        Popul[i] = new double[NVars];
        for(int j=0;j!=NVars;j++)
            Popul[i][j] = Random(Left,Right);
    }
}
Optimizer::~Optimizer()
{
    for(int i=0;i!=NInds;i++)
        delete Popul[i];
    delete Popul;
    delete Fitness;
    delete Trial;
    delete BestHistory;
}
void Optimizer::Run(GNBG& gnbg)
{
    for(int i=0;i!=NInds;i++)
    {
        Fitness[i] = gnbg.Fitness(Popul[i]);
        BestF = (i == 0)*Fitness[i] + (i != 0)*min(BestF,Fitness[i]);
        if(gnbg.FEval%1 == 0)
        {
            BestHistory[BestHistoryIndex] = BestF;
            BestHistoryIndex++;
        }
    }
    while(gnbg.FEval < gnbg.MaxEvals)
    {
        for(int i=0;i!=NInds;i++)
        {
            int r1 = IntRandom(NInds);
            int r2 = IntRandom(NInds);
            int r3 = IntRandom(NInds);
            int jrand = IntRandom(NVars);
            double RV;
            F = Random(0.4,1.0);
            Cr = Random(0.0,1.0);
            for(int j=0;j!=NVars;j++)
            {
                RV = Random(0.0,1.0);
                bool cros = (RV < Cr || j == jrand);
                //if first should crossover then perform mutation, else take target vector component
                Trial[j] = cros*(Popul[r1][j] + F*(Popul[r2][j] - Popul[r3][j])) + (!cros)*Popul[i][j];
            }
            tempfit = gnbg.Fitness(Trial);
            bool repl = (tempfit <= Fitness[i]);
            //if better then replace, else keep the same
            for(int j=0;j!=NVars;j++)
                Popul[i][j] = repl*Trial[j] + (!repl)*Popul[i][j];
            Fitness[i] = repl*tempfit + (!repl)*Fitness[i];
            BestF = min(BestF,Fitness[i]);
            if(gnbg.FEval%1 == 0)
            {
                BestHistory[BestHistoryIndex] = BestF;
                BestHistoryIndex++;
            }
            if(gnbg.FEval == gnbg.MaxEvals)
                break;
        }
    }
}
int main()
{
    int mode = 1;
        /*0 - generate function, save parameters to file, test random points;
          1 - generate function, save parameters to file, generate grid and evaluate, save for visualization (change Dimension to 2!)
          2 - generate function, run differential evolution
        */
    if(mode == 0)
    {
        int N = 10000;
        GNBG gnbg;
        gnbg.Init();
        gnbg.save_to_file("func.txt");
        double* xvec = new double[gnbg.Dimension];
        for(int i=0;i!=N;i++)
        {
            cout<<i<<"\t";
            for(int j=0;j!=gnbg.Dimension;j++)
            {
                xvec[j] = Random(gnbg.MinCoordinate,gnbg.MaxCoordinate);
                cout<<xvec[j]<<"\t";
            }
            double f = gnbg.Fitness(xvec);
            cout<<f<<endl;
        }
        gnbg.Clear();
        delete xvec;
    }
    else if(mode == 1) //change Dimension to 2!
    {
        GNBG gnbg;
        gnbg.Init();
        gnbg.save_to_file("func.txt");
        double* xvec = new double[gnbg.Dimension];
        ofstream fout("vis.txt");
        int N = 101;
        double step = (gnbg.MaxCoordinate - gnbg.MinCoordinate)/double(N-1);
        double** fvals = new double*[N];
        for(int i=0;i!=N;i++)
            fvals[i] = new double[N];
        for(int i=0;i!=N;i++)
        {
            for(int j=0;j!=N;j++)
            {
                xvec[0] = gnbg.MinCoordinate + step*i;
                xvec[1] = gnbg.MinCoordinate + step*j;
                fvals[i][j] = gnbg.Fitness(xvec);
                fout<<fvals[i][j]<<"\t";
            }
            fout<<endl;
        }
        gnbg.Clear();
        delete xvec;
        for(int i=0;i!=N;i++)
            delete fvals[i];
        delete fvals;
    }
    else if(mode == 2)
    {
        GNBG gnbg;
        gnbg.Init();
        Optimizer Opt(100,gnbg);
        Opt.Run(gnbg);
        char buffer[100];
        sprintf(buffer,"Res_DE_.txt");
        ofstream fout_c(buffer);
        fout_c.precision(20);
        for(int i=0;i!=Opt.BestHistoryIndex;i++)
        {
            fout_c<<Opt.BestHistory[i]-gnbg.OptimumValue<<"\n";
        }
        fout_c.close();
        gnbg.Clear();
    }
    return 0;
}
