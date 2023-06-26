
all_filenames = []
directory = "Resources/TerrorismEventData/test-ans"
directory2 = "Responses/colab_test_chunks/"

# directory = "Temp/test"
# directory2 = "Temp/our_ans/"

with open("filenames_test.txt", 'r') as filenames:
    for line in filenames:
        line = line.strip()
        all_filenames.append(line)
    filenames.close()

# all_filenames.append("TEST_1")

def get_correctly_labelled(anskey_list, our_ans_list):
    # print(anskey_list)
    # print(our_ans_list)
    correctly_labelled = 0
    for our_ans in our_ans_list:
        #Remove after use?
        #'-' case?
        for i in range(len(anskey_list)):
            anskey_set = anskey_list[i]
            if our_ans in anskey_set:
                correctly_labelled += 1
                break
    return correctly_labelled

def add_multiline_input(curr_list, line):
    true_instance_labelled = 0
    curr_set = set()
    if '/' in line:
        for each_item in line.split('/'):
            if len(each_item.strip()) == 0:
                continue
            # print(each_item.strip())
            curr_set.add(each_item.strip())
        curr_list.append(curr_set)
        true_instance_labelled += 1
    elif len(line.strip()) != 0:
        curr_set.add(line.strip())
        curr_list.append(curr_set)
        true_instance_labelled += 1
    return true_instance_labelled

def get_score(correctly_labelled, system_labelled, answerkey_labelled):
    precision = 0
    recall = 0
    fscore = 0
    if system_labelled != 0:
        precision = correctly_labelled / system_labelled
    if answerkey_labelled != 0:
        recall = correctly_labelled / answerkey_labelled
    if (precision + recall != 0):
        fscore = (2 * precision * recall) / (precision + recall)
    return (recall, precision, fscore)

correctly_labelled_incident = 0
true_instance_incident = 0
all_system_labelled_incident = 0

correctly_labelled_weapon = 0
true_instance_weapon = 0
all_system_labelled_weapon = 0

correctly_labelled_perp = 0
true_instance_perp = 0
all_system_labelled_perp = 0

correctly_labelled_victim = 0
true_instance_victim = 0
all_system_labelled_victim = 0

correctly_labelled_target = 0
true_instance_target = 0
all_system_labelled_target = 0

for filename in all_filenames:
    anskey_incident = ""
    our_answer_incident = ""

    all_anskey_weapon = []
    our_answer_weapon = []

    all_anskey_perp = []
    our_answer_perp = []

    all_anskey_victim = []
    our_answer_victim = []

    all_anskey_target = []
    our_answer_target = []

    # if filename != "DEV-MUC3-0795":
    #     continue
    # print(filename)
    with open(directory + "/" + filename + ".anskey", 'r') as answer_key:
        is_perp = False
        is_weapon = False
        is_target = False
        is_victim = False
        
        for line in answer_key:
            line = line.strip()
            if "INCIDENT:" in line:
                anskey_incident = line.split(':')[1].strip()
                true_instance_incident += 1
            elif "WEAPON:" in line: 
                line = line.split(':')[1].strip()
                curr_weapon = set()
                if len(line) == 0:
                    continue
                elif len(line) == 1 and '-' in line:
                    continue
                #     curr_weapon.add('-')
                #     all_anskey_weapon.append(curr_weapon)
                #     true_instance_weapon += 1 
                #     continue
                elif '/' in line:
                    for each_weapon in line.split('/'):
                        curr_weapon.add(each_weapon.strip())
                    true_instance_weapon += 1
                    all_anskey_weapon.append(curr_weapon)
                    is_weapon = True
                else:
                    curr_weapon.add(line.strip())
                    true_instance_weapon += 1
                    all_anskey_weapon.append(curr_weapon)
                    is_weapon = True
            elif "PERP " in line:
                is_weapon = False
                org = line.split(':')[1].strip()
                if "PERP INDIV:" in line:
                    if '-' in org:
                        continue
                if len(org.strip()) == 0:
                    continue
                if '-' in org and len(all_anskey_perp) != 0:
                    continue
                if '-' in org and len(org) == 1:
                    # curr_perp = set()
                    # curr_perp.add('-')
                    # all_anskey_perp.append(curr_perp)
                    # true_instance_perp += 1 
                    continue
                elif '/' in org:
                    curr_perp = set()
                    for each_org in org.split(' / '):
                        curr_perp.add(each_org.strip())
                    true_instance_perp += 1 
                    all_anskey_perp.append(curr_perp)
                    is_perp = True
                else:
                    perp_2 = set()
                    true_instance_perp += 1 
                    perp_2.add(org.strip())
                    all_anskey_perp.append(perp_2)
                    is_perp = True                
            elif is_weapon:
                true_instance_weapon += add_multiline_input(all_anskey_weapon, line)
            elif "TARGET:" in line:
                is_perp = False
                line = line.split(':')[1].strip()
                if len(line.strip()) == 0:
                    continue
                elif len(line) == 1 and '-' in line:
                    # curr_target = set()
                    # curr_target.add('-')
                    # all_anskey_target.append(curr_target)
                    # true_instance_target += 1 
                    continue
                elif '/' in line:
                    curr_target = set()
                    for each_org in line.split(' / '):
                        curr_target.add(each_org.strip())
                    true_instance_target += 1 
                    all_anskey_target.append(curr_target)
                    is_target = True
                else:
                    curr_target = set()
                    true_instance_target += 1 
                    curr_target.add(line.strip())
                    all_anskey_target.append(curr_target)
                    is_target = True
                continue
            elif is_perp:
                true_instance_perp += add_multiline_input(all_anskey_perp, line)
            elif "VICTIM:" in line:
                is_target = False
                line = line.split(':')[1].strip()
                if len(line.strip()) == 0:
                        continue
                anskey_victim = set()
                if len(line) == 1 and '-' in line:
                    # anskey_victim.add('-')
                    # all_anskey_victim.append(anskey_victim)
                    # true_instance_victim += 1
                    continue
                elif '/' in line:
                    for each_org in line.split(' / '):
                        if len(each_org.strip()) == 0:
                            continue
                        anskey_victim.add(each_org.strip())
                    true_instance_victim += 1
                    all_anskey_victim.append(anskey_victim)
                    is_victim = True
                else:
                    if (len(line) == 0):
                        continue
                    anskey_victim.add(line.strip())
                    all_anskey_victim.append(anskey_victim)
                    true_instance_victim += 1
                    is_victim = True
            elif is_target:
                true_instance_target += add_multiline_input(all_anskey_target, line)
            elif is_victim:
                true_instance_victim += add_multiline_input(all_anskey_victim, line)   
    answer_key.close()
    
    with open(directory2 + "/" + filename, 'r') as our_answer:
        is_perp = False
        is_weapon = False
        is_target = False
        is_victim = False
        for line in our_answer:
            if len(line.strip()) == 0:
                continue 
            if "INCIDENT:" in line:
                our_answer_incident = line.split(':')[1].strip()
                all_system_labelled_incident += 1
            elif "WEAPON:" in line: 
                our_answer_weapon_curr = line.split(':')[1].strip()
                if len(our_answer_weapon_curr) == 1 and '-' in our_answer_weapon_curr:
                    # our_answer_weapon.append('-')
                    # all_system_labelled_weapon += 1
                    continue
                else:
                    is_weapon = True
                    our_answer_weapon.append(our_answer_weapon_curr.strip())
                    all_system_labelled_weapon += 1
            
            elif "PERP:" in line:
                is_weapon = False
                our_answer_perp_curr = line.split(':')[1].strip()
                if len(our_answer_perp_curr) == 1 and '-' in our_answer_perp_curr:
                        # our_answer_perp.append('-')
                        # all_system_labelled_perp += 1
                        continue
                else:
                    our_answer_perp.append(our_answer_perp_curr.strip())
                    all_system_labelled_perp += 1
                    is_perp = True
            elif is_weapon:
                our_answer_weapon.append(line.strip())
                all_system_labelled_weapon += 1
            elif "VICTIM:" in line:
                is_perp = False
                our_answer_victim_curr = line.split(':')[1].strip()
                if len(our_answer_victim_curr) == 1 and '-' in our_answer_victim_curr:
                        # our_answer_victim.append('-')
                        # all_system_labelled_victim += 1
                        continue
                else:
                    our_answer_victim.append(our_answer_victim_curr.strip())
                    all_system_labelled_victim += 1
                    is_victim = True
            elif is_perp:
                our_answer_perp.append(line.strip())
                all_system_labelled_perp += 1
            elif "TARGET:" in line:
                is_victim = False
                our_answer_target_curr = line.split(':')[1].strip()
                if len(our_answer_target_curr) == 1 and '-' in our_answer_target_curr:
                    # our_answer_target.append('-')
                    # all_system_labelled_target += 1
                    continue
                else:
                    our_answer_target.append(our_answer_target_curr.strip())
                    all_system_labelled_target += 1
                    is_target = True
            elif is_victim:
                our_answer_victim.append(line.strip())
                all_system_labelled_victim += 1
            elif is_target:
                line = line.strip()
                if (len(line) == 0):
                        continue
                else:
                    our_answer_target.append(line)
                    all_system_labelled_target += 1

    our_answer.close()

    if(anskey_incident == our_answer_incident):
        # print("ID: " + filename + "\nANSWER KEY: " + anskey_incident + "\tOUR_ANSWER: " + our_answer_incident)
    # else:
        correctly_labelled_incident += 1
  
    correctly_labelled_weapon += get_correctly_labelled(all_anskey_weapon, our_answer_weapon)
    correctly_labelled_perp += get_correctly_labelled(all_anskey_perp, our_answer_perp)
    correctly_labelled_victim += get_correctly_labelled(all_anskey_victim, our_answer_victim)
    correctly_labelled_target += get_correctly_labelled(all_anskey_target, our_answer_target)


print("INCIDENT")
incident_score = get_score(correctly_labelled_incident, all_system_labelled_incident, true_instance_incident)
print("Recall: ", incident_score[1])
print("Precision: ", incident_score[2])
print("F Score: ", incident_score[0])
print()

print("WEAPON")
weapon_score = get_score(correctly_labelled_weapon, all_system_labelled_weapon, true_instance_weapon)
print("Recall: ", weapon_score[0])
print("Precision: ", weapon_score[1])
print("F Score: ", weapon_score[2])

print()
print("PERP")
perp_score = get_score(correctly_labelled_perp, all_system_labelled_perp, true_instance_perp)
print("Recall: ", perp_score[0])
print("Precision: ", perp_score[1])
print("F Score: ", perp_score[2])
print()

print("VICTIM")
victim_score = get_score(correctly_labelled_victim, all_system_labelled_victim, true_instance_victim)
print("Recall: ", victim_score[0])
print("Precision: ", victim_score[1])
print("F Score: ", victim_score[2])
print()

print("TARGET")
target_score = get_score(correctly_labelled_target, all_system_labelled_target, true_instance_target)
print("Recall: ", target_score[0])
print("Precision: ", target_score[1])
print("F Score: ", target_score[2])
# print()
