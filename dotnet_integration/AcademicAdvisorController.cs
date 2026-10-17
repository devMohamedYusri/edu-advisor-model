using System.Collections.Generic;
using Microsoft.AspNetCore.Mvc;
using AcademicAdvisor.AI;

namespace YourProject.Controllers
{
    public class RecommendationRequestDto
    {
        public List<string> PassedCourses { get; set; } = new List<string>();
        public List<CourseItem> AvailableCourses { get; set; } = new List<CourseItem>();
        public int MaxCreditHours { get; set; } = 9;
    }

    [ApiController]
    [Route("api/v1/recommend")]
    public class AcademicAdvisorController : ControllerBase
    {
        private readonly AcademicAdvisorEngine _advisorEngine;

        // Injected via Dependency Injection (Program.cs)
        public AcademicAdvisorController(AcademicAdvisorEngine advisorEngine)
        {
            _advisorEngine = advisorEngine;
        }

        [HttpPost]
        public IActionResult GetRecommendation([FromBody] RecommendationRequestDto request)
        {
            if (request == null || request.AvailableCourses == null || request.AvailableCourses.Count == 0)
            {
                return BadRequest(new { success = false, message = "available_courses cannot be empty." });
            }

            var result = _advisorEngine.RecommendRegistrationBundle(
                request.PassedCourses,
                request.AvailableCourses,
                request.MaxCreditHours
            );

            return Ok(result);
        }
    }
}
