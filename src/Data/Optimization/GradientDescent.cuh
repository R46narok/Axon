#ifndef _AXON_GRADIENT_DESCENT
#define _AXON_GRADIENT_DESCENT

#include "Core/Library.cuh"

namespace Axon
{
    class AXON_API NeuralNetwork;
    class AXON_API Matrix;

    class AXON_API GradientDescent
    {
    public:
        explicit GradientDescent(uint32_t iterations = 1500, float learningRate = 0.25f);

        void Optimize(NeuralNetwork& network, const Matrix& input, const Matrix& output) const;

        [[nodiscard]] inline float GetLearningRate() const noexcept { return m_LearningRate; }
        [[nodiscard]] inline uint32_t GetIterations() const noexcept { return m_Iterations; }
    private:
        float m_LearningRate;
        uint32_t m_Iterations;
    };
}

#endif //_AXON_GRADIENT_DESCENT
