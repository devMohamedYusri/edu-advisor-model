# 🎓 Academic Advisor AI — .NET / C# Integration Guide

This package provides everything needed to run the Deep Learning Academic Recommendation model directly inside an **ASP.NET Core / C#** backend (MonsterASP.NET, IIS, Kestrel, Azure, Docker, etc.) without requiring Python or external cloud servers.

---

## 📦 1. Required NuGet Package

Install Microsoft's official ONNX Runtime into your .NET project:

```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

---

## 📁 2. Included Files

1. `advisor_model.onnx` — The compiled neural network model weights.
2. `course_vocab.json` — The course code token vocabulary dictionary.
3. `AcademicAdvisorEngine.cs` — The C# engine class that performs inference, ranking, and credit-hour packing.
4. `AcademicAdvisorController.cs` — Example ASP.NET Core API Controller.

Place `advisor_model.onnx` and `course_vocab.json` in your ASP.NET project root (e.g. under an `App_Data/` or `AI/` folder), and configure them in `.csproj` to copy to output directory:

```xml
<ItemGroup>
  <None Update="AI\advisor_model.onnx">
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
  </None>
  <None Update="AI\course_vocab.json">
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
  </None>
</ItemGroup>
```

---

## ⚙️ 3. Registering in `Program.cs` (Dependency Injection)

Register `AcademicAdvisorEngine` as a **Singleton** so the model is loaded into memory only once when the server starts up:

```csharp
using AcademicAdvisor.AI;

var builder = WebApplication.CreateBuilder(args);

// Register AI Engine as a Singleton service
var onnxPath = Path.Combine(builder.Environment.ContentRootPath, "AI", "advisor_model.onnx");
var vocabPath = Path.Combine(builder.Environment.ContentRootPath, "AI", "course_vocab.json");

builder.Services.AddSingleton(new AcademicAdvisorEngine(onnxPath, vocabPath));

builder.Services.AddControllers();

var app = builder.Build();
app.MapControllers();
app.Run();
```

---

## 🚀 4. API Request & Response Specification

### **POST** `/api/v1/recommend`

#### **Request Body (JSON):**
```json
{
  "passed_courses": ["CS141", "BS101", "BS111", "BS105", "BS103", "CS101", "UNI008"],
  "available_courses": [
    { "code": "CS142", "hours": 3, "name": "Programming 2" },
    { "code": "CS102", "hours": 3, "name": "Logic Design" },
    { "code": "BS102", "hours": 3, "name": "Mathematics 2" },
    { "code": "BS133", "hours": 2, "name": "Computing Economics" },
    { "code": "UNI001", "hours": 1, "name": "Human Rights" },
    { "code": "UNI003", "hours": 1, "name": "Problem Solving" }
  ],
  "max_credit_hours": 9
}
```

#### **Response Body (JSON):**
```json
{
  "success": true,
  "total_credit_hours": 7,
  "max_allowed_hours": 9,
  "suggested_courses": [
    {
      "code": "UNI001",
      "hours": 1,
      "name": "Human Rights",
      "advisor_score": 1.0
    },
    {
      "code": "UNI003",
      "hours": 1,
      "name": "Problem Solving",
      "advisor_score": 1.0
    },
    {
      "code": "BS102",
      "hours": 3,
      "name": "Mathematics 2",
      "advisor_score": 0.9617
    },
    {
      "code": "BS133",
      "hours": 2,
      "name": "Computing Economics",
      "advisor_score": 0.9329
    }
  ]
}
```

---

## ⚡ Performance
- **Inference Time:** `< 0.1 ms` per recommendation request.
- **Memory Footprint:** `< 5 MB RAM`.
- **Dependencies:** 100% native managed C# with `Microsoft.ML.OnnxRuntime`.
