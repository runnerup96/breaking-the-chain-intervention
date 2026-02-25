from copy import deepcopy
from datasets_for_intervention.prompt import Prompt


class RiceChemIntervention:
    def __init__(self, dataset, llm_model, tool, processor, prompting_regime='standard', tool_mode=None):
        self.dataset = dataset
        self.llm_model = llm_model

        assert prompting_regime in ["standard", "detailed", "max_detailed"]
        assert tool_mode in [None, 'simple', 'structured']

        self.prompting_regime = prompting_regime

        self.tool = tool
        self.processor = processor
        self.tool_mode = tool_mode

    def clean_llm_output(self, text):
        tokens_to_remove = ['<|im_end|>',
                            '<|endoftext|>',
                            '<|im_start|>',
                            '<|eot_id|>',
                            '<|pad|>',
                            '\u00ad',
                            '\u200b',
                            '\u200c',
                            '\u200d',
                            '\u2060',
                            '\ufeff']

        for token in tokens_to_remove:
            text = text.replace(token, '')
        return text.strip()

    def infer_completion(self, completion: str, sample: dict = None, short_completion: bool = False):
        if self.tool_mode:
            tool_args = self.processor.extract_tool_args(completion, short_completion)
            if self.tool_mode == "structured":
                tool_rubric = self.processor.boollist_to_checklist(sample, tool_args)
            else:
                tool_rubric = tool_args
            score = self.tool.calculate_score({"rubric": tool_args}, sample)
            return score, tool_rubric
        final_grade = self.processor.extract_final_answer(completion, short_completion)
        return final_grade, {}

    def make_intervention(self, sample: dict, generated_output: dict):
        completion = self.clean_llm_output(generated_output['completion'])
        sample['raw_generation'] = completion

        if sample["completion_type"] == "structure_prediction":
            mediator_rubric = self.processor.extract_mediator(completion)
            sample["mediator_rubric"] = mediator_rubric or {}

        short_completion = sample["completion_type"] == "gold_structure"
        sample['score_before_intervention'], tool_rubric = self.infer_completion(completion, sample, short_completion)
        sample['tool_rubric'] = tool_rubric

        interventions = self.make_structure_intervention(sample)
        sample['structure_intervention'] = interventions
        return sample

    def make_structure_intervention(self, sample: dict):
        """HSVT + Local + Correction (in gold_structure regime)"""
        task_idx = sample['task_idx']

        # HSVT
        hsvt = deepcopy(sample)
        hsvt['student_answer'] = self.dataset.get_random_student_answer(task_idx)

        # Local Edits
        local_edits = []
        for item, answer in sample['mediator_rubric'].items():
            local = deepcopy(sample)
            local['mediator_rubric'][item] = not answer
            local['expected_score_after_intervention'] = self.tool.calculate_score({"rubric": local['mediator_rubric']}, local)
            local_edits.append(local)

        # Correction
        correction = []
        if sample['completion_type'] == 'gold_structure' and isinstance(sample.get("bad_rubric"), dict) and len(sample["bad_rubric"]) > 0:
            corr = deepcopy(sample)
            corr['mediator_rubric'] = corr['bad_rubric']
            correction = [corr]

        return {
            "HSVT": [hsvt],
            "Local Edits": local_edits,
            "Correction": correction
        }

    def interventions_to_prompt(self, sample: dict):
        interv = sample['structure_intervention']
        prompts = []
        prompts += [self.make_prompt(interv['HSVT'][0], include_gold_structure=True)]
        prompts += [self.make_prompt(edit, include_gold_structure=True) for edit in interv['Local Edits']]
        if sample['completion_type'] == 'gold_structure' and len(interv['Correction']) > 0:
            prompts += [self.make_prompt(interv['Correction'][0], include_gold_structure=True)]
        return prompts

    def collect_intervention_completion(self, sample: dict, generated_output: list):
        completions = [self.clean_llm_output(g['completion']) for g in generated_output]
        interv = sample['structure_intervention']
        idx = 0

        interv['HSVT'][0]['score_after_intervention'], tool_rubric = self.infer_completion(completions[idx], interv['HSVT'][0], short_completion=True)
        if self.tool_mode:
            interv['HSVT'][0]['tool_rubric_after_intervention'] = tool_rubric
        idx += 1
        for i in range(len(interv['Local Edits'])):
            interv['Local Edits'][i]['score_after_intervention'], tool_rubric = self.infer_completion(completions[idx], interv['Local Edits'][i], short_completion=True)
            if self.tool_mode:
                interv['Local Edits'][i]['tool_rubric_after_intervention'] = tool_rubric
            idx += 1

        if sample['completion_type'] == 'gold_structure' and len(interv['Correction']) > 0:
            interv['Correction'][0]['score_after_intervention'], tool_rubric = self.infer_completion(completions[idx], interv['Correction'][0], short_completion=True)
            if self.tool_mode:
                interv['Correction'][0]['tool_rubric_after_intervention'] = tool_rubric

        return sample

    
    FEW_SHOT = {
        "Example 1": {"Question": ("When studying the emission sources within the Milky Way, a satellite detected interplanetary clouds containing silicon atoms that have lost five electrons.\n"
                                    "b) The ionization energies corresponding to the removal of the third, fourth, and fifth electrons in silicon are 3231, 4356, and 16091 kJ/mol, respectively. \n"
                                    "Using core charge calculations and your understanding of Coulomb's Law, briefly explain 1) why the removal of each additional electron requires more energy than the removal of the previous one, and 2) the relative magnitude of the values observed.\n"
                                    "This question can be answered reasonably in around 150 words or fewer.\n"), 
                        "Answer": ("With each removal of an electron, there is less electron-electron repulsion, which decreases the potential energy of the electrons as they are more strongly attracted to the nucleus, and ultimately increasing each successive ionization energy.  "
                                    "The ionization energies of the third and fourth electron are similar due to the fact that both of these electrons reside in the same n quantum number (3), meaning they are basically the same radius away from the nucleus. Furthermore, these two electrons have the same core charge of +4. This indicates the potential energies and thus the resulting ionization energies are similar, as Coulomb's Law states potential energy is given by V(r) =(+Ze)(-e)/r. The difference in these two energies is due to the fact that the electrons in the 3p orbital experience greater electron-electron repulsion than those in the 3s, and 3s electrons have greater probability of core penetration. This is supported by silicon's electron configuration of 1s^2 2s^2 2p^6 3s^2 3p^2.  "
                                    "However, there is a large jump in ionization energy from removal of the fourth to fifth electron because there is a significant decrease in the distance between the electron and nucleus (r), as the fifth electron is removed from the n=2 shell instead of the third. Thus, the core charge felt by the fifth electron is +12, significantly increasing the ionization energy.\n"),
                        "Checklist": {"correctly cites decreased electron electron repulsion (True/False)": True,
                                    "relates decreased electron electron repulsion to decreased potential energy (True/False)": True,
                                    "3rd and 4th electrons ionized feel same core charge (True/False)": True,
                                    "3rd and 4th electrons ionized from n=3 shell and have same radius (True/False)": True,
                                    "5th electron ionized from n=2 shell and feels higher core charge (True/False)": True,
                                    "5th electron ionized from n=2 shell and has smaller radius (True/False)": True,
                                    "correctly explains relationship of potential energy to ionization energy (True/False)": True,
                                    "partially explains relationship between potential energy and ionization energy (True/False)": False},
                        "Explanation": "There are no interventions here"
                        
                        },
        "Example 2": {"Question": ("In each statement below (a-c), two observations are given which seem to contrast with each other. Using your knowledge of electron configurations, orbitals, Coulomb’s law, and/or atomic and molecular structures, briefly explain why both of these observations are true, and how the two observations can be reconciled in each case.\n\n"
                                    "b) If light is used to excite an electron to a higher energy level in an atom, only certain frequencies of light can be absorbed. However, if it is used to eject an electron from the atom, any value above a minimum threshold frequency can be absorbed. What’s up with that?! ¯\\ (°-°) /¯\n\n"
                                    "This question can be answered reasonably in around 150 words or fewer.\n"),
                        "Answer": ("The reason why only certain frequencies of light can excite electrons to a higher energy level in an atom is because the energy levels that the electron will go to match the energy levels of that specific frequency of light."
                                    "Think about it, if the energy level above the one that the electron is currently is like let's say -6 eV and the one that the electron is at is at like -10 eV, then the electron will be needed to be hit with a frequency of light that is 4 eV to get to the -6 eV."
                                    "If it is not exactly 4 then it wont be able to catch on to that energy level. However when you are ejecting an electron, you are not trying to reach a specific energy level, you are just trying to get out of the atom,"
                                    "so the frequency that you need is the frequency required to get out of the atom Once the electron is out of the atom, it is out. So the frequency does not really matter after that point that the electron is out of the atom. It is just an added bonus. The frequency of the light correlates with its energy, especially kinetic energy. The more the frequency the faster it will go. The threshold frequency is simply how much energy the electron needs to break free from the prison of the atom. If the electron has more energy than it needs, then it does not matter and it will continue to break free."),
                        "Checklist": {"Correctly states that frequency is proportional to energy of light (True/False)": False,
                                    "Explaining sentence 1: energy levels of an electron in an atom are quantized (True/False)": True,
                                    "Explaining sentence 1: FULLY explains energy/frequency absorbed must equal the difference in energy levels in an electron (True/False)": True,
                                    "Explaining sentence 1: PARTIALLY explains energy/frequency absorbed must equal the difference in energy levels in an electron (True/False)": False,
                                    "Explaining sentence 2: a minimum amount of energy is needed to eject an electron (True/False)": False,
                                    "Explaining sentence 2: any additional energy becomes kinetic energy (True/False)": True
                                    },
                        "Explanation": "There are no interventions here"
                        },
        "Example 3": {"Question": ("A CHEM 121 student was asked what hybrid orbitals must be present to form methanimine (CH2NH), for which a correct Lewis structure is shown below:\n\n"
                                    "The student responded:\n"
                                    "According to valence bond theory, Carbon cannot form four bonds because it only has two unpaired valence electrons. So, it has to form four sp3 hybrid orbitals to create the four bonds. Nitrogen doesn’t need to hybridize because it already has three unpaired 2p valence electrons to form the three bonds with Carbon and Hydrogen.\n"
                                    "Assess the accuracy and logic of the student’s response: briefly explain whether the reasoning presented is logical, noting what information is correct or incorrect and providing correct logical reasoning and explanation where needed.\n"
                                    "This question can be reasonably answered in 150 words or fewer."),
                        "Answer": ("Sentence 1: This is incorrect, valence bond theory dictates that carbon cannot form 4 bonds because its valence electrons only occupy 3 atomic orbitals, one 2s and two 2p orbitals, and therefore atomic orbital overlap would only account for Carbon having three bonds."
                                    "Sentence 2: This is not correct, while carbon has 4 bonds it only has 3 electron domains around it and therefore undergoes sp^2 hybridization to form three sp^2 orbitals. Two of these orbitals form the single bonds with H while the remaining sp^2 orbital alongside a pi bond created between the unhybridized 2p orbitals in carbon and nitrogen form a double bond."
                                    "Sentence 3: This is incorrect, Nitrogen does in fact undergo sp^2 hybridization as it has three electron domains around it. One of the three sp^2 orbitals facilitates the single N-H bond while another sp^2 orbital in conjuction with a remaning 2p orbital in the same plane of carbon's 2p form a double bond between nitrogen and carbon."),
                        "Checklist": {"Sentence 1 is correct. Valence bond theory describes that atomic orbitals must be half-filled to participate in covalent bonding. (True/False)": False,
                                    "Sentence 2: Correct number of hybrid orbitals. In this molecule, carbon must form three hybrid orbitals to form three electron domains. (True/False)": True,
                                    "Sentence 2: Correct type of hybrid orbitals. Carbon must form sp2 hybrid orbitals (from using a 2s and two 2p orbitals) (True/False)": True,
                                    "Sentence 3: Correctly states that nitrogen is hybridized (True/False)": True,
                                    "Sentence 3: Correct type of hybridization. Nitrogen is sp2 hybridized to form 3 electron domains (True/False)": True,
                                    "Sentence 3: Correct description of hybrid orbital bonds in nitrogen. Two sp2 orbitals form two sigma bonds. (True/False)": True,
                                    "Sentence 3: Correct description of unhybridized orbital bonds in nitrogen. Unhybridized p orbital forms pi bond (True/False)": True
                                    },
                        "Checklist_with_intervention": {"Sentence 1 is correct. Valence bond theory describes that atomic orbitals must be half-filled to participate in covalent bonding. (True/False)": True,
                                                        "Sentence 2: Correct number of hybrid orbitals. In this molecule, carbon must form three hybrid orbitals to form three electron domains. (True/False)": True,
                                                        "Sentence 2: Correct type of hybrid orbitals. Carbon must form sp2 hybrid orbitals (from using a 2s and two 2p orbitals) (True/False)": False,
                                                        "Sentence 3: Correctly states that nitrogen is hybridized (True/False)": True,
                                                        "Sentence 3: Correct type of hybridization. Nitrogen is sp2 hybridized to form 3 electron domains (True/False)": False,
                                                        "Sentence 3: Correct description of hybrid orbital bonds in nitrogen. Two sp2 orbitals form two sigma bonds. (True/False)": False,
                                                        "Sentence 3: Correct description of unhybridized orbital bonds in nitrogen. Unhybridized p orbital forms pi bond (True/False)": True
                                                    },
                        "Explanation": "Here the original score must be 6.0, but after intervention it should have been recalculated to 4.0 (because of 4 True's).",                    
                        },
        "Example 4": {"Question": ("How did the Law of Multiple Proportions lead to the conclusion that matter is made of atoms?\n"
                                    "This question can be reasonably answered in around 75 words or fewer.\n"),
                        "Answer": ("The Law of Multiple Proportions states that when two elements combine to form more than one compound, if one of the elements is fixed to a certain mass in each compound, the mass of the other element will exist in a simple integer ratio to the masses of that element in the other compounds."
                                    "The appearance of a simple integer ratio implies that something is being counted, and that being the smallest divisible unit. As this is mass data, that means this must be a unit of mass, which was concluded to be the atom, with molecules being made up of a whole number sum of them."),
                        "Checklist": {"Fixed mass of one element (True/False)": True,
                                    "Mass data in LoMP (True/False)": True,
                                    "Combine to form compounds (True/False)": True,
                                    "Integer/whole number ratio (True/False)": True,
                                    "Whole numbers mean indivisible/discrete (True/False)": True,
                                    "Indivisible unit of mass = atom (True/False)": True
                                    },
                        "Checklist_with_intervention": {"Fixed mass of one element (True/False)": False,
                                                        "Mass data in LoMP (True/False)": False,
                                                        "Combine to form compounds (True/False)": False,
                                                        "Integer/whole number ratio (True/False)": False,
                                                        "Whole numbers mean indivisible/discrete (True/False)": False,
                                                        "Indivisible unit of mass = atom (True/False)": False
                                                        },
                        "Explanation": "Here the original score must be 6.0, but after intervention it should have been recalculated to 0.0 (because of 0 True's)."
                        }
    }

    def _checklist_dict_to_string(self, checklist: dict) -> str:
        if not isinstance(checklist, dict) or not checklist:
            return ""
        lines = []
        for k, v in checklist.items():
            lines.append(f"{k}: {'True' if v else 'False'}")
        return "\n".join(lines)

    def _get_tool_call_string(self, checklist: dict) -> str:
        if not self.tool_mode:
            return ""

        if self.tool_mode == "simple":
            filled = self._checklist_dict_to_string(checklist)
            s = filled.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            args = f'{{"rubric": "{s}"}}'
            return "Final tool call:\nTOOL: calculate_score\nARGS: " + args + "\n\n"

        bools = []
        for _, v in checklist.items():
            bools.append(True if v is True else False)
        items = ", ".join(["True" if b else "False" for b in bools])
        args = f'{{"rubric": [{items}]}}'
        return "Final tool call:\nTOOL: calculate_score\nARGS: " + args + "\n\n"

    def make_prompt(self, ricechem_sample: dict, include_gold_structure: bool = False) -> str:
        checklist = []
        # item2weight = self.dataset.task2rubric_weights[ricechem_sample['task_idx']]
        for rubric_item in ricechem_sample['gold_rubric']:
            # checklist_item = f"{rubric_item} (weight: {item2weight[rubric_item]}) (True/False): <True/False>\n"
            checklist_item = f"{rubric_item} (True/False): <True/False>\n"
            checklist.append(checklist_item)
        checklist_string = "".join(checklist)

        instruction = (
            "You are an automated grader for a college-level chemistry class. "
            "Your task is to evaluate a student's answer by first constructing a structured reasoning block "
            "(a checklist of reasoning steps with weights) and then compute a final grade.\n\n"
            "Task explanation:\n"
            "- You are given a question, a student's answer, and a checklist of rubric items.\n"
            "- You must fill the checklist (True/False) strictly based on the student's answer.\n"
            "- The final grade equals the number of the items marked True.\n\n"
            "Intermediate structure construction (Checklist):\n"
            "- Use only the given question and student's answer—do not assume or invent new items.\n"
            "- Keep the checklist text EXACTLY as provided (same order and wording). "
            "Only replace the trailing <True/False> with True or False for each line.\n"
            "- Mark an item True only if the student's answer explicitly satisfies it; otherwise mark False.\n"
            "- If the checklist contains mutually exclusive items (e.g., FULLY vs PARTIALLY), never mark both True.\n\n"
        )

        if not self.tool_mode:
            standard_output_rule = (
                "After filling the checklist, compute the final grade as the number of True items."
                "Express the grade as a float.\n\n"

                "Important output rule:\n"
                "Your final response must contain ONLY two fields and no other text:\n"
                "1) Checklist: (the filled checklist, line-for-line in the same format)\n"
                "2) Final grade: <float>\n\n"
            )

            instruction += standard_output_rule

        
        tool_call_instruction = ""
        if self.tool_mode == "simple":
            tool_call_instruction = (
                "Tool usage (REQUIRED):\n"
                "- After you fill the checklist, you MUST call the tool to compute the final grade.\n"
                "- Tool name: calculate_score\n"
                "- IMPORTANT: The tool input must be the RAW filled rubric/checklist in EXACTLY the same format that you generated.\n"
                "- Provide ARGS as valid JSON. Escape newlines in the rubric string as \\n.\n\n"

                "Important output rule:\n"
                "Your final response must contain ONLY the following fields and no other text:\n"
                "1) Checklist: (the filled checklist, line-for-line in the same format)\n"
                "2) Final tool call:\n"
                "   TOOL: calculate_score\n"
                "   ARGS: {\"rubric\": \"FILLED RICECHEM CHECKLIST\"}\n\n"
                "IMPORTANT: in the final answer, you only have to call the tool, do NOT try to count the score yourself and do NOT output it after the Final Grade.\n\n"
            )
        elif self.tool_mode == "structured":
            tool_call_instruction = (
                "Tool usage (REQUIRED):\n"
                "- After you fill the checklist, you MUST call the tool to compute the final grade.\n"
                "- Tool name: calculate_score\n"
                "- IMPORTANT: tool input is a boolean list aligned with your checklist lines:\n"
                "  * same ORDER as the checklist lines\n"
                "  * same LENGTH as the checklist lines\n"
                "  * element i corresponds to checklist line i\n"
                "- Do NOT compute the grade yourself.\n\n"

                "Important output rule:\n"
                "Your final response must contain ONLY the following fields and no other text:\n"
                "1) Checklist: (the filled checklist, line-for-line in the same format)\n"
                "2) Final tool call:\n"
                "   TOOL: calculate_score\n"
                "   ARGS: {\"rubric\": [True, False, ...]}\n\n"
                "IMPORTANT: in the final answer, you only have to call the tool, do NOT try to count the score yourself and do NOT output it after the Final Grade.\n\n"
            )

        tool_spec_block = ""
        if self.tool_mode:
            tool_spec_block = "Tool specification:\n" + self.tool.spec_json() + "\n\n"

        tool_call_instruction += tool_spec_block

        few_shot_text = "FEW-SHOT EXAMPLES:\n\n"
        for ex_name in ["Example 1", "Example 2", "Example 3", "Example 4"]:
            ex = self.FEW_SHOT[ex_name]
            ex_num = ex_name.split()[-1]

            if self.prompting_regime == "detailed":
                checklist = ex["Checklist_with_intervention"] if "Checklist_with_intervention" in ex else ex["Checklist"]
                explanation = ex["Explanation"]
            else:
                checklist = ex["Checklist"]
                explanation = ""

            checklist_str = self._checklist_dict_to_string(checklist)

            ex_block = (
                f"Example #{ex_num}\n"
                "Question:\n"
                f"{ex['Question']}"
                "Answer:\n"
                f"{ex['Answer']}\n"
                "Checklist:\n"
                f"{checklist_str}\n"
            )

            if not self.tool_mode:
                score = float(sum(1 for _, v in checklist.items() if v is True))
                ex_block += f"Final grade ({ricechem_sample['score_range']}): {score:.1f}\n\n"
            else:
                ex_block += self._get_tool_call_string(checklist)

            if self.prompting_regime in ["detailed", "max_detailed"]:
                ex_block += "Explanation:\n" + explanation + "\n"

            few_shot_text += ex_block

        current_sample = (
            "Now follow the same structure for the given input.\n\n"
            "Question:\n"
            f"{ricechem_sample['task']}\n\n"
            "Answer:\n"
            f"{ricechem_sample['student_answer']}\n\n"
            "Checklist:\n"
            f"{checklist_string}"
        )

        prompt = Prompt(
            prompting_regime=self.prompting_regime,
            use_tool_call=self.tool_mode is not None,
            tool_call_instruction=tool_call_instruction,
            instruction=instruction,
            few_shot=few_shot_text,
            llm_model=self.llm_model,
        )

        gold_structure = None
        if include_gold_structure:
            gold_structure = "Checklist:\n"
            for rubric_item, answer in ricechem_sample['mediator_rubric'].items():
                # checklist_item = f"{rubric_item} (weight: {item2weight[rubric_item]}) (True/False): {answer}\n"
                gold_structure += f"{rubric_item} (True/False): {answer}\n"
            #checklist_string += "Final grade (0-8): "
            if self.tool_mode:
                gold_structure += "Final tool call:\n"
            else:
                gold_structure += f"Final grade ({ricechem_sample['score_range']}): "

        return prompt.make_prompt(
            current_sample=current_sample,
            include_gold_structure=include_gold_structure,
            gold_structure=gold_structure,
        )
