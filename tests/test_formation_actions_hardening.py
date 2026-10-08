from discrete.models import ClaimDraft
from discrete.process.pipeline import ProcessPipeline


async def test_similar_actions_with_different_recipients_stay_separate(store):
    await store.create_claim(ClaimDraft(kind="event", text="The user sent the report to Alice", action="sent",
                                      occurred_from=100, occurred_to=105))
    cd = ClaimDraft(kind="event", text="The user sent the report to Bob", action="sent",
                    occurred_from=101, occurred_to=106)
    assert (await ProcessPipeline(store)._action_write(cd)).operation == "added"


async def test_different_amounts_and_files_stay_separate(store):
    await store.create_claim(ClaimDraft(kind="event", text="The user paid $40 to Acme", action="paid",
                                      occurred_from=100, occurred_to=105))
    cd = ClaimDraft(kind="event", text="The user paid $45 to Acme", action="paid", occurred_from=101, occurred_to=106)
    assert (await ProcessPipeline(store)._action_write(cd)).operation == "added"


async def test_a_more_specific_wording_of_one_action_reinforces(store):
    await store.create_claim(ClaimDraft(kind="event", text="The user downloaded the Helium browser",
                                      action="downloaded", occurred_from=100, occurred_to=105))
    cd = ClaimDraft(kind="event", text="The user downloaded the Helium browser installer helium_0.18.dmg",
                    action="downloaded", occurred_from=200, occurred_to=205)
    assert (await ProcessPipeline(store)._action_write(cd)).operation == "reinforced"
