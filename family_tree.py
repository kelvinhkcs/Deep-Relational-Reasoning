"""Implement family tree generator and family tree class."""

import random as ran
import jacinle.random as random
import numpy as np
import json
import openai
from sklearn.metrics import f1_score
import os
from openai import AzureOpenAI, OpenAI

__all__ = ['Family', 'randomly_generate_family']


class Family(object):
    """Family tree class to support queries about relations between family members.

    Args:
    nr_people: The number of people in the family tree.
    relations: The relations between family members. The relations should be an
    matrix of shape [nr_people, nr_people, 6]. The relations in the order
    are: husband, wife, father, mother, son, daughter.
    """

    def __init__(self, nr_people, relations):
        self._n = nr_people
        self._relations = relations

    def mul(self, x, y):
        return np.clip(np.matmul(x, y), 0, 1)

    @property
    def nr_people(self):
        return self._n

    @property
    def relations(self):
        return self._relations

    @property
    def father(self):
        return self._relations[:, :, 2]

    @property
    def mother(self):
        return self._relations[:, :, 3]

    @property
    def son(self):
        return self._relations[:, :, 4]

    @property
    def daughter(self):
        return self._relations[:, :, 5]

    def has_father(self):
        return self.father.max(axis=1)

    def has_mother(self):
        return self.mother.max(axis=1)

    def has_daughter(self):
        return self.daughter.max(axis=1)

    def has_sister(self):
        daughter_cnt = self.daughter.sum(axis=1)
        is_daughter = np.clip(self.daughter.sum(axis=0), 0, 1)
        return ((np.matmul(self.father, daughter_cnt) - is_daughter) > 0).astype('float')

    def get_mothers(self):
        return np.clip(self.mother, 0, 1)

    def get_fathers(self):
        return np.clip(self.father, 0, 1)

    def get_daughters(self):
        return np.clip(self.daughter, 0, 1)

    def get_sons(self):
        return np.clip(self.son, 0, 1)

    def get_parents(self):
        return np.clip(self.father + self.mother, 0, 1)

    def get_grandfather(self):
        return self.mul(self.get_parents(), self.father)

    def get_grandmother(self):
        return self.mul(self.get_parents(), self.mother)

    def get_children(self):
        return np.clip(self.daughter + self.son, 0, 1)

    def get_grandsons(self):
        return self.mul(self.get_children(), self.son)

    def get_grandparents(self):
        parents = self.get_parents()
        return self.mul(parents, parents)

    def get_uncle(self):
        return np.clip(self.mul(self.get_grandparents(), self.son) - self.father, 0, 1)

    def get_aunt(self):
        return np.clip(self.mul(self.get_grandparents(), self.daughter) - self.mother, 0, 1)

    def get_maternal_great_uncle(self):
        return self.mul(self.mul(self.get_grandmother(), self.mother), self.son)

    def get_paternal_great_aunt(self):
        return self.mul(np.clip(self.mul(self.get_grandfather(), self.get_parents()), 0, 1), self.daughter)


def randomly_generate_family(n, p_marriage=0.9, verbose=False):
    """Randomly generate family trees.

    Mimic the process of families growing using a timeline. Each time a new person
    is created, randomly sample the gender and parents (could be none, indicating
    not included in the family tree) of the person. Also maintain lists of singles
    of each gender. With probability $p_marrige, randomly pick two from each list
    to be married. Finally randomly permute the order of people.

    Args:
    n: The number of people in the family tree.
    p_marriage: The probability of marriage happens each time.
    verbose: print the marriage and child born process if verbose=True.
    Returns:
    A family tree instance of $n people.
    """
    assert n > 0
    ids = list(random.permutation(n))

    single_m = []
    single_w = []
    couples = [None]
    # The relations are: husband, wife, father, mother, son, daughter
    rel = np.zeros((n, n, 6))
    fathers = [None for i in range(n)]
    mothers = [None for i in range(n)]

    def add_couple(man, woman):
        """Add a couple relation among (man, woman)."""
        couples.append((man, woman))
        rel[woman, man, 0] = 1  # husband
        rel[man, woman, 1] = 1  # wife
        if verbose:
            print('couple', man, woman)

    def add_child(parents, child, gender):
        """Add a child relation between parents and the child according to gender."""
        father, mother = parents
        fathers[child] = father
        mothers[child] = mother
        rel[child, father, 2] = 1  # father
        rel[child, mother, 3] = 1  # mother
        if gender == 0:  # son
            rel[father, child, 4] = 1
            rel[mother, child, 4] = 1
        else:  # daughter
            rel[father, child, 5] = 1
            rel[mother, child, 5] = 1
        if verbose:
            print('child', father, mother, child, gender)

    def check_relations(man, woman):
        """Disable marriage between cousins."""
        if fathers[man] is None or fathers[woman] is None:
            return True
        if fathers[man] == fathers[woman]:
            return False

        def same_parent(x, y):
            return fathers[x] is not None and fathers[y] is not None and fathers[x] == fathers[y]

        for x in [fathers[man], mothers[man]]:
            for y in [fathers[woman], mothers[woman]]:
                if same_parent(man, y) or same_parent(woman, x) or same_parent(x, y):
                    return False
        return True

    while ids:
        x = ids.pop()
        gender = random.randint(2)
        #     print(couples)
        parents = ran.choice(couples)
        #     print(parents)
        if gender == 0:
            single_m.append(x)
        else:
            single_w.append(x)
        if parents is not None:
            add_child(parents, x, gender)

        if random.rand() < p_marriage and len(single_m) > 0 and len(single_w) > 0:
            mi = random.randint(len(single_m))
            wi = random.randint(len(single_w))
            man = single_m[mi]
            woman = single_w[wi]
            if check_relations(man, woman):
                add_couple(man, woman)
                del single_m[mi]
                del single_w[wi]

    return Family(n, rel)

def generate_family(n, p_marriage):
    fam = randomly_generate_family(n, p_marriage)
    fathers_arr = fam.get_fathers()
    mothers_arr = fam.get_mothers()
    daughters_arr = fam.get_daughters()
    sons_arr = fam.get_sons()
    sisters_arr = fam.has_sister()
    grandsons_arr = fam.get_grandsons()
    aunts_arr = fam.get_aunt()
    paternal_great_aunts_arr = fam.get_paternal_great_aunt()
    return fathers_arr, mothers_arr, daughters_arr, sons_arr, sisters_arr, grandsons_arr, aunts_arr, paternal_great_aunts_arr

def generate_relations(n, fathers_arr, mothers_arr, daughters_arr, sons_arr):
    statements_arr = []

    for i in range(n):
        for j in range(n):
            if fathers_arr[i,j] == 1:
                statements_arr.append("P"+str(j)+" is P"+str(i)+"'s father.")
            if mothers_arr[i,j] == 1:
                statements_arr.append("P"+str(j)+" is P"+str(i)+"'s mother.")
            if daughters_arr[i,j] == 1:
                statements_arr.append("P"+str(j)+" is P"+str(i)+"'s daughter.")
            if sons_arr[i,j] == 1:
                statements_arr.append("P"+str(j)+" is P"+str(i)+"'s son.")
                
    ran.shuffle(statements_arr)

    return ' '.join(statements_arr)

def create_system_prompt():
    return """
    The user will provide a question performing reasoning on a family tree.
    Please give some reasoning steps briefly and give your matrix in a list of list in JSON format. 

    EXAMPLE JSON OUTPUT:
    {
        "Brief Reasoning Steps": "Some brief reasoning steps.",
        "Matrix": [[1, 2, 3], [-1, 2, 3], [1, -1, 2]],"
    }
    """

def create_user_prompt_sisters(n, relations):
    return "You are an agent who determines the relations in a family. \
    For a family tree containing " + str(n) + " family members, \
    which is depicted with 4 kinds of relations: father, mother, son, daughter. \
    The relations are: " + relations + \
    " Now, from the above-given facts, you have to determine who has relation sister. \
    You must give the reasoning process and you must give the final answer in a the format of a " + str(n) + " 1d-matrix, \
    where the i-th entry is 1 Pj has sister(s), 0 otherwise. \
    Question: who has sister(s)? \
    I need the matrix for further processing. Do not include anything other than the matrix in python-readable format in 'Matrix'.\
    Your answer shall be in JSON format. \
    Answer: "

def create_user_prompt_grandsons(n, relations):
    return "You are an agent who determines the relations in a family. \
    For a family tree containing " + str(n) + " family members, \
    which is depicted with 4 kinds of relations: father, mother, son, daughter. \
    The relations are: " + relations + \
    " Now, from the above-given facts, you have to determine the relation grandsons. \
    You must give the reasoning process and you must give the final answer in a " + str(n) + "-by-" + str(n) + " matrix, \
    where the i,j-th entry is 1 if Pj is Pi's grandson, 0 otherwise. \
    Question: who has the relation grandson? \
    I need the matrix for further processing. Do not include anything other than the matrix in python-readable format in 'Matrix'.\
    Your answer shall be in JSON format. \
    Answer: "

def create_user_prompt_aunts(n, relations):
    return "You are an agent who determines the relations in a family. \
    For a family tree containing " + str(n) + " family members, \
    which is depicted with 4 kinds of relations: father, mother, son, daughter. \
    The relations are: " + relations + \
    " Now, from the above-given facts, you have to determine the relation aunt. \
    You must give the reasoning process and you must give the final answer in a " + str(n) + "-by-" + str(n) + " matrix, \
    where the i,j-th entry is 1 if Pj is Pi's aunt, 0 otherwise. \
    Question: who has the relation aunt? \
    I need the matrix for further processing. Do not include anything other than the matrix in python-readable format in 'Matrix'.\
    Your answer shall be in JSON format. \
    Answer: "

def create_user_prompt_paternal_great_aunts(n, relations):
    return "You are an agent who determines the relations in a family. \
    For a family tree containing " + str(n) + " family members, \
    which is depicted with 4 kinds of relations: father, mother, son, daughter. \
    The relations are: " + relations + \
    " Now, from the above-given facts, you have to determine the relation paternal great aunt. \
    You must give the reasoning process and you must give the final answer in a " + str(n) + "-by-" + str(n) + " matrix, \
    where the i,j-th entry is 1 if Pj is Pi's paternal great aunt, 0 otherwise. \
    Question: who has the relation paternal great aunt? \
    I need the matrix for further processing. Do not include anything other than the matrix in python-readable format in 'Matrix'.\
    Your answer shall be in JSON format. \
    Answer: "

def create_client_ds(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

def create_client_gpt(api_key, api_version, azure_endpoint):
    return AzureOpenAI(api_version=api_version, api_key=api_key, azure_endpoint=azure_endpoint)

def prompt_llm_ds(client, sys_prompt, user_prompt, llm_model = "deepseek-chat"):
    response = client.chat.completions.create(
        model=llm_model,  
        max_tokens=8192,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        response_format={
            'type': 'json_object'
        }
    )
    return response

def prompt_llm_ds_r1(client, sys_prompt, user_prompt, llm_model = "deepseek-reasoner"):
    response = client.chat.completions.create(
        model=llm_model,  
        max_tokens=8192,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    return response

def prompt_llm_gpt_4o(client, sys_prompt, user_prompt, llm_model = "gpt-4o"):
    response = client.chat.completions.create(
        model=llm_model,  
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    return response

def extract_json_and_numpy_array(llm_output):
    response = llm_output.choices[0].message.content
    response = response.replace("```json", "")
    response = response.replace("```", "")
    return json.loads(response), np.array(json.loads(response)['Matrix'])

def cal_f1_score(correct_matrix, output_matrix, weight_method='micro'):
    return f1_score(correct_matrix.ravel(), output_matrix.ravel(), average=weight_method)

def save_json_numpy(file_name, json_output, output_matrix):
    with open(file_name+'.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
        