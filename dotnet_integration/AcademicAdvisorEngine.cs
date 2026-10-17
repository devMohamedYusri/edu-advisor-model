using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AcademicAdvisor.AI
{
    public class CourseItem
    {
        public string Code { get; set; } = string.Empty;
        public int Hours { get; set; } = 3;
        public string Name { get; set; } = string.Empty;
        public float AdvisorScore { get; set; }
    }

    public class RecommendationResult
    {
        public bool Success { get; set; } = true;
        public int TotalCreditHours { get; set; }
        public int MaxAllowedHours { get; set; }
        public List<CourseItem> SuggestedCourses { get; set; } = new List<CourseItem>();
    }

    public class AcademicAdvisorEngine : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly Dictionary<string, long> _vocab;

        public AcademicAdvisorEngine(string onnxModelPath, string vocabJsonPath)
        {
            if (!File.Exists(onnxModelPath))
                throw new FileNotFoundException($"ONNX model file not found at: {onnxModelPath}");

            if (!File.Exists(vocabJsonPath))
                throw new FileNotFoundException($"Vocabulary JSON file not found at: {vocabJsonPath}");

            _session = new InferenceSession(onnxModelPath);

            var jsonContent = File.ReadAllText(vocabJsonPath);
            _vocab = JsonSerializer.Deserialize<Dictionary<string, long>>(jsonContent) 
                     ?? new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        }

        public RecommendationResult RecommendRegistrationBundle(
            List<string> passedCourses,
            List<CourseItem> availableCourses,
            int maxCreditHours)
        {
            if (availableCourses == null || availableCourses.Count == 0)
            {
                return new RecommendationResult
                {
                    Success = false,
                    MaxAllowedHours = maxCreditHours,
                    TotalCreditHours = 0,
                    SuggestedCourses = new List<CourseItem>()
                };
            }

            // 1. Encode student passed courses to vocabulary IDs
            var normalizedPassed = passedCourses ?? new List<string>();
            var historyIds = normalizedPassed
                .Select(c => c.Trim().ToUpperInvariant())
                .Where(c => _vocab.ContainsKey(c))
                .Select(c => _vocab[c])
                .DefaultIfEmpty(0L)
                .ToArray();

            // 2. Encode candidate courses to vocabulary IDs
            var candidateIds = availableCourses
                .Select(c => {
                    var code = (c.Code ?? string.Empty).Trim().ToUpperInvariant();
                    return _vocab.TryGetValue(code, out var id) ? id : 0L;
                })
                .ToArray();

            // 3. Create Tensors (Shape: [1, history_length] and [1, num_candidates])
            var historyTensor = new DenseTensor<long>(historyIds, new[] { 1, historyIds.Length });
            var candidateTensor = new DenseTensor<long>(candidateIds, new[] { 1, candidateIds.Length });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("history_ids", historyTensor),
                NamedOnnxValue.CreateFromTensor("candidate_ids", candidateTensor)
            };

            // 4. Run Neural Network Inference (< 0.1 ms in C#)
            using var results = _session.Run(inputs);
            var scoresTensor = results.First().AsTensor<float>();

            var evaluatedList = new List<CourseItem>();
            for (int i = 0; i < availableCourses.Count; i++)
            {
                var candidate = availableCourses[i];
                var score = scoresTensor[0, i];
                evaluatedList.Add(new CourseItem
                {
                    Code = candidate.Code,
                    Hours = candidate.Hours,
                    Name = candidate.Name,
                    AdvisorScore = (float)Math.Round(score, 4)
                });
            }

            // 5. Rank by Advisor Score (Descending) and Pack Credit Budget
            var rankedCandidates = evaluatedList.OrderByDescending(c => c.AdvisorScore).ToList();
            var bundle = new List<CourseItem>();
            int accumulatedHours = 0;

            foreach (var course in rankedCandidates)
            {
                if (accumulatedHours + course.Hours <= maxCreditHours)
                {
                    bundle.Add(course);
                    accumulatedHours += course.Hours;
                    if (accumulatedHours == maxCreditHours)
                        break;
                }
            }

            return new RecommendationResult
            {
                Success = true,
                TotalCreditHours = accumulatedHours,
                MaxAllowedHours = maxCreditHours,
                SuggestedCourses = bundle
            };
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
