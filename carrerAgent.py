import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

from langgraph.graph import Graph, StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Set up your Anthropic API key
# os.environ["ANTHROPIC_API_KEY"] = "your_api_key_here"

class CareerStage(Enum):
    STUDENT = "student"
    EARLY_CAREER = "early_career"
    MID_CAREER = "mid_career"
    SENIOR_CAREER = "senior_career"
    CAREER_CHANGE = "career_change"

@dataclass
class UserProfile:
    name: str
    age: int
    education: str
    current_role: Optional[str]
    experience_years: int
    skills: List[str]
    interests: List[str]
    career_stage: CareerStage
    goals: List[str]
    preferences: Dict[str, Any]

class SkillAssessment(BaseModel):
    technical_skills: Dict[str, int] = Field(description="Technical skills rated 1-10")
    soft_skills: Dict[str, int] = Field(description="Soft skills rated 1-10")
    skill_gaps: List[str] = Field(description="Areas needing improvement")
    strengths: List[str] = Field(description="Key strengths")
    recommendations: List[str] = Field(description="Skill development recommendations")

class CareerRecommendation(BaseModel):
    job_titles: List[str] = Field(description="Recommended job titles")
    industries: List[str] = Field(description="Suitable industries")
    salary_range: str = Field(description="Expected salary range")
    growth_potential: str = Field(description="Career growth potential")
    reasoning: str = Field(description="Reasoning for recommendations")
    next_steps: List[str] = Field(description="Immediate next steps")

class DevelopmentPlan(BaseModel):
    short_term_goals: List[str] = Field(description="3-6 month goals")
    medium_term_goals: List[str] = Field(description="6-18 month goals")
    long_term_goals: List[str] = Field(description="1-3 year goals")
    learning_resources: List[Dict[str, str]] = Field(description="Recommended learning resources")
    action_items: List[str] = Field(description="Specific action items")
    timeline: str = Field(description="Suggested timeline")

class CareerGuidanceState:
    """State object that gets passed between nodes in the graph"""
    
    def __init__(self):
        self.messages: List[BaseMessage] = []
        self.user_profile: Optional[UserProfile] = None
        self.skill_assessment: Optional[SkillAssessment] = None
        self.career_recommendations: Optional[CareerRecommendation] = None
        self.development_plan: Optional[DevelopmentPlan] = None
        self.current_step: str = "profile_collection"
        self.conversation_history: List[Dict] = []

class CareerGuidanceAgent:
    def __init__(self, anthropic_api_key: str):
        """Initialize the career guidance agent"""
        self.llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key=anthropic_api_key,
            temperature=0.1
        )
        self.skill_parser = JsonOutputParser(pydantic_object=SkillAssessment)
        self.career_parser = JsonOutputParser(pydantic_object=CareerRecommendation)
        self.plan_parser = JsonOutputParser(pydantic_object=DevelopmentPlan)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for career guidance"""
        
        # Define the state schema
        def add_message_to_state(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
            return left + right
        
        # Create the graph
        workflow = StateGraph(CareerGuidanceState)
        
        # Add nodes
        workflow.add_node("profile_collector", self._collect_user_profile)
        workflow.add_node("skill_assessor", self._assess_skills)
        workflow.add_node("career_recommender", self._recommend_careers)
        workflow.add_node("plan_developer", self._develop_plan)
        workflow.add_node("advisor", self._provide_guidance)
        
        # Define the flow
        workflow.set_entry_point("profile_collector")
        workflow.add_edge("profile_collector", "skill_assessor")
        workflow.add_edge("skill_assessor", "career_recommender")
        workflow.add_edge("career_recommender", "plan_developer")
        workflow.add_edge("plan_developer", "advisor")
        workflow.add_edge("advisor", END)
        
        return workflow.compile()
    
    def _collect_user_profile(self, state: CareerGuidanceState) -> CareerGuidanceState:
        """Collect and structure user profile information"""
        
        profile_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career counselor collecting user information. 
            Based on the user's input, extract and structure their profile information.
            Ask follow-up questions if important information is missing.
            
            Focus on gathering:
            - Personal details (name, age, education)
            - Current role and experience
            - Skills and interests
            - Career goals and preferences
            - Career stage (student, early career, mid career, senior, career change)
            """),
            ("human", "{user_input}")
        ])
        
        # This would typically involve interactive conversation
        # For demo purposes, we'll create a sample profile
        sample_profile = UserProfile(
            name="Sample User",
            age=28,
            education="Bachelor's in Computer Science",
            current_role="Software Developer",
            experience_years=3,
            skills=["Python", "JavaScript", "SQL", "Problem Solving", "Communication"],
            interests=["Machine Learning", "Data Science", "Leadership"],
            career_stage=CareerStage.EARLY_CAREER,
            goals=["Advance to senior role", "Learn ML/AI", "Increase salary"],
            preferences={"work_style": "remote", "company_size": "medium", "industry": "tech"}
        )
        
        state.user_profile = sample_profile
        state.current_step = "skill_assessment"
        
        return state
    
    def _assess_skills(self, state: CareerGuidanceState) -> CareerGuidanceState:
        """Assess user's current skills and identify gaps"""
        
        skill_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career counselor conducting a skill assessment.
            Based on the user's profile, provide a comprehensive skill evaluation.
            
            Rate technical and soft skills on a scale of 1-10.
            Identify skill gaps and areas for improvement.
            Provide specific recommendations for skill development.
            
            Return your response as a JSON object matching the SkillAssessment schema.
            """),
            ("human", """User Profile:
            Name: {name}
            Current Role: {current_role}
            Experience: {experience_years} years
            Skills: {skills}
            Interests: {interests}
            Career Goals: {goals}
            
            Please provide a detailed skill assessment.""")
        ])
        
        chain = skill_prompt | self.llm | self.skill_parser
        
        result = chain.invoke({
            "name": state.user_profile.name,
            "current_role": state.user_profile.current_role,
            "experience_years": state.user_profile.experience_years,
            "skills": ", ".join(state.user_profile.skills),
            "interests": ", ".join(state.user_profile.interests),
            "goals": ", ".join(state.user_profile.goals)
        })
        
        state.skill_assessment = SkillAssessment(**result)
        state.current_step = "career_recommendations"
        
        return state
    
    def _recommend_careers(self, state: CareerGuidanceState) -> CareerGuidanceState:
        """Provide career recommendations based on skills and interests"""
        
        career_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career counselor providing career recommendations.
            Based on the user's profile and skill assessment, recommend suitable career paths.
            
            Consider:
            - Current skills and experience
            - Career interests and goals
            - Industry trends and growth potential
            - Salary expectations
            - Career progression opportunities
            
            Return your response as a JSON object matching the CareerRecommendation schema.
            """),
            ("human", """User Profile: {profile}
            
            Skill Assessment:
            Technical Skills: {technical_skills}
            Soft Skills: {soft_skills}
            Strengths: {strengths}
            Skill Gaps: {skill_gaps}
            
            Please provide detailed career recommendations.""")
        ])
        
        chain = career_prompt | self.llm | self.career_parser
        
        result = chain.invoke({
            "profile": f"Role: {state.user_profile.current_role}, Experience: {state.user_profile.experience_years} years, Goals: {', '.join(state.user_profile.goals)}",
            "technical_skills": str(state.skill_assessment.technical_skills),
            "soft_skills": str(state.skill_assessment.soft_skills),
            "strengths": ", ".join(state.skill_assessment.strengths),
            "skill_gaps": ", ".join(state.skill_assessment.skill_gaps)
        })
        
        state.career_recommendations = CareerRecommendation(**result)
        state.current_step = "development_planning"
        
        return state
    
    def _develop_plan(self, state: CareerGuidanceState) -> CareerGuidanceState:
        """Develop a personalized career development plan"""
        
        plan_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career counselor creating a development plan.
            Based on the user's profile, skill assessment, and career recommendations,
            create a detailed, actionable career development plan.
            
            Include:
            - Short, medium, and long-term goals
            - Specific learning resources and courses
            - Action items with timelines
            - Skill development priorities
            
            Return your response as a JSON object matching the DevelopmentPlan schema.
            """),
            ("human", """User Profile: {profile}
            
            Career Recommendations: {recommendations}
            Skill Gaps: {skill_gaps}
            
            Create a comprehensive development plan.""")
        ])
        
        chain = plan_prompt | self.llm | self.plan_parser
        
        result = chain.invoke({
            "profile": f"Current: {state.user_profile.current_role}, Goals: {', '.join(state.user_profile.goals)}",
            "recommendations": f"Suggested roles: {', '.join(state.career_recommendations.job_titles)}",
            "skill_gaps": ", ".join(state.skill_assessment.skill_gaps)
        })
        
        state.development_plan = DevelopmentPlan(**result)
        state.current_step = "guidance"
        
        return state
    
    def _provide_guidance(self, state: CareerGuidanceState) -> CareerGuidanceState:
        """Provide final guidance and advice"""
        
        guidance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior career counselor providing final guidance.
            Synthesize all the information gathered to provide comprehensive career advice.
            
            Be encouraging, specific, and actionable in your recommendations.
            Address potential challenges and how to overcome them.
            """),
            ("human", """Based on the complete career analysis, provide final guidance and next steps for the user.""")
        ])
        
        chain = guidance_prompt | self.llm
        
        guidance = chain.invoke({})
        
        state.messages.append(HumanMessage(content=guidance.content))
        state.current_step = "complete"
        
        return state
    
    def run_career_guidance(self, user_input: str = None) -> Dict[str, Any]:
        """Run the complete career guidance workflow"""
        
        # Initialize state
        initial_state = CareerGuidanceState()
        if user_input:
            initial_state.messages.append(HumanMessage(content=user_input))
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Prepare results
        results = {
            "user_profile": final_state.user_profile.__dict__ if final_state.user_profile else None,
            "skill_assessment": final_state.skill_assessment.dict() if final_state.skill_assessment else None,
            "career_recommendations": final_state.career_recommendations.dict() if final_state.career_recommendations else None,
            "development_plan": final_state.development_plan.dict() if final_state.development_plan else None,
            "guidance": final_state.messages[-1].content if final_state.messages else None
        }
        
        return results
    
    def get_interactive_guidance(self, user_query: str) -> str:
        """Provide interactive career guidance for specific queries"""
        
        interactive_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career counselor providing personalized advice.
            Answer the user's specific career-related question with actionable guidance.
            Be supportive, knowledgeable, and provide concrete next steps.
            """),
            ("human", "{query}")
        ])
        
        chain = interactive_prompt | self.llm
        response = chain.invoke({"query": user_query})
        
        return response.content

# Example usage
def main():
    # Initialize the agent (you'll need to provide your Anthropic API key)
    # agent = CareerGuidanceAgent("your_anthropic_api_key")
    
    # For demo purposes, we'll show how to use the agent
    print("Career Guidance Agent Example")
    print("=" * 40)
    
    # Example of running the complete workflow
    # results = agent.run_career_guidance("I'm a software developer looking to advance my career")
    
    # Example of interactive guidance
    # response = agent.get_interactive_guidance("How can I transition from software development to data science?")
    
    print("Agent initialized successfully!")
    print("To use the agent:")
    print("1. Set your ANTHROPIC_API_KEY environment variable")
    print("2. Install required dependencies: langgraph, langchain-anthropic, pydantic")
    print("3. Create an agent instance and call run_career_guidance() or get_interactive_guidance()")

if __name__ == "__main__":
    main()

# Additional utility functions for the career guidance agent

class CareerDatabase:
    """Simple career database for storing and retrieving career information"""
    
    def __init__(self):
        self.career_data = {
            "data_scientist": {
                "skills": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"],
                "salary_range": "$80k-$150k",
                "growth_rate": "High",
                "education": "Bachelor's in STEM field preferred"
            },
            "product_manager": {
                "skills": ["Communication", "Analytics", "Strategy", "Leadership", "User Research"],
                "salary_range": "$90k-$160k",
                "growth_rate": "High",
                "education": "Bachelor's degree, MBA preferred"
            },
            "software_engineer": {
                "skills": ["Programming", "Problem Solving", "System Design", "Testing", "Version Control"],
                "salary_range": "$70k-$180k",
                "growth_rate": "Very High",
                "education": "Bachelor's in Computer Science or related field"
            }
        }
    
    def get_career_info(self, career: str) -> Dict[str, Any]:
        """Get career information from the database"""
        return self.career_data.get(career.lower().replace(" ", "_"), {})
    
    def search_careers_by_skill(self, skill: str) -> List[str]:
        """Find careers that require a specific skill"""
        matching_careers = []
        for career, info in self.career_data.items():
            if skill.lower() in [s.lower() for s in info.get("skills", [])]:
                matching_careers.append(career.replace("_", " ").title())
        return matching_careers

# Integration with external APIs (example structure)
class CareerAPIIntegration:
    """Integration with external career APIs for real-time data"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def get_salary_data(self, job_title: str, location: str) -> Dict[str, Any]:
        """Get salary data from external API"""
        # This would integrate with APIs like Glassdoor, PayScale, etc.
        return {
            "median_salary": "$95,000",
            "salary_range": "$75k-$120k",
            "location": location,
            "data_source": "API Integration"
        }
    
    def get_job_market_trends(self, field: str) -> Dict[str, Any]:
        """Get job market trends"""
        # This would integrate with job market APIs
        return {
            "demand": "High",
            "growth_projection": "15% over next 5 years",
            "hot_skills": ["AI/ML", "Cloud Computing", "Data Analysis"],
            "field": field
        }

# Resume analysis component
class ResumeAnalyzer:
    """Analyze resumes and provide improvement suggestions"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_resume(self, resume_text: str, target_role: str) -> Dict[str, Any]:
        """Analyze resume against target role"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional resume reviewer. Analyze the provided resume
            for the target role and provide detailed feedback on:
            - Strengths and weaknesses
            - Missing keywords
            - Format improvements
            - Content suggestions
            - ATS optimization tips
            """),
            ("human", """Resume Text: {resume}
            Target Role: {role}
            
            Provide detailed analysis and recommendations.""")
        ])
        
        chain = analysis_prompt | self.llm
        response = chain.invoke({"resume": resume_text, "role": target_role})
        
        return {"analysis": response.content, "target_role": target_role}