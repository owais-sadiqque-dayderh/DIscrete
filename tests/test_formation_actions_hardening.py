from discrete.models import ClaimDraft
from discrete.process.pipeline import ProcessPipeline


async def test_similar_actions_with_different_recipients_stay_separate(store):
    await store.create_claim(ClaimDraft(kind="event", text="The user sent the report to Alice", action="sent",
                                      occurred_from=100, occurred_to=105))
    cd = ClaimDraft(kind="event", text="The user sent the report to Bob", action="sent",
                    occurred_from=101, occurred_to=106)
    assert (await ProcessPipeline(store)._action_write(cd)).operation == "added"


async def test_identical_actions_without_shared_occurrence_stay_separate(store):
    cd = ClaimDraft(kind="event", text="The user sent the report to Alice", action="sent",
                    occurred_from=100, occurred_to=105)
    await store.create_claim(cd)
    assert (await ProcessPipeline(store)._action_write(cd)).operation == "added"
