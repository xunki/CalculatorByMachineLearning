using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Test
{
    class Program
    {
        private class Parameter
        {
            public float X { get; set; }
            public float Y { get; set; }
            public float Result { get; set; }
        }

        private class Result
        {
            [ColumnName("Score")]
            public float Data { get; set; }
        }

        private static void Main(string[] args)
        {
            Calc("+", (x, y) => x + y);
            Calc("-", (x, y) => x - y);
            Calc("*", (x, y) => x * y);
            Calc("/", (x, y) => x / y);
        }

        private static void Calc(string operatorName, Func<float, float, float> func)
        {
            Console.WriteLine($"======= 当前实现运算符 {operatorName} =======");

            var test = GenerateDemoData(func, 10000);

            // 模型训练
            var mlContext = new MLContext();
            var demoData = mlContext.Data.LoadFromEnumerable(test);
            var model = mlContext.Transforms.Concatenate("Features", "X", "Y")
                // 使用 LightGbm 算法，如果使用 Sdca 成功率无法达到 100%
                .Append(mlContext.Regression.Trainers.LightGbm("Result"))
                .Fit(demoData);
            var metrics = mlContext.Regression.Evaluate(model.Transform(demoData), "Result");
            Console.WriteLine(
                $"运算符 {operatorName} 模型训练完成，准确度：{metrics.RSquared:P} 误差系数：{metrics.RootMeanSquaredError:P3}");

            // 验证数据
            var engine = mlContext.Model.CreatePredictionEngine<Parameter, Result>(model);
            var verifyData = GenerateDemoData(func, 5);
            foreach (var parameter in verifyData)
            {
                // [验证精度] 因为 float 精度确实，显示的值可能不精准，所以在一定精度范围内将认为是精确的值
                const int PRECISION = 1;

                var result = Math.Round(engine.Predict(parameter).Data, PRECISION);
                var funcResult = Math.Round(parameter.Result, PRECISION);

                Console.WriteLine(
                    $"{(Math.Abs(result - funcResult) < Math.Pow(10, -PRECISION + 1) ? "√" : "×")} | {parameter.X} {operatorName} {parameter.Y} = {result}");
            }
        }

        private static List<Parameter> GenerateDemoData(Func<float, float, float> func, int size)
        {
            var random = new Random();
            return Enumerable.Range(1, size).Select(i =>
            {
                const int MAX_NUM = 10;
                var p = new Parameter
                {
                    X = random.Next(1, MAX_NUM),
                    Y = random.Next(1, MAX_NUM)
                };
                p.Result = func(p.X, p.Y);
                return p;
            }).ToList();
        }
    }
}